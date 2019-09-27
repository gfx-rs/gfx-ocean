#![cfg_attr(
    not(any(feature = "vulkan", feature = "dx12", feature = "metal")),
    allow(dead_code, unused_extern_crates, unused_imports)
)]

#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use gfx_hal as hal;
use nalgebra_glm as glm;

use std::{fs, io::Read, time::Instant};

use crate::hal::{
    command::{ClearColor, ClearDepthStencil, ClearValue},
    format::{ChannelType, Format, Swizzle},
    buffer as b, command, format as f, image as i, memory as m, pass, pool, pso,
    DescriptorPool, Device, IndexType, Instance, PhysicalDevice, Primitive,
    Submission, Surface, Swapchain, SwapchainConfig,
    window::Extent2D
};

use winit::dpi::PhysicalSize;

#[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
use ocean::{CorrectionLocals, PropagateLocals};

mod camera;
#[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
mod fft;
#[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
mod ocean;

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
struct Vertex {
    a_Pos: [f32; 3],
    a_Uv: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
struct PatchOffset {
    a_offset: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
struct Locals {
    a_proj: [[f32; 4]; 4],
    a_view: [[f32; 4]; 4],
    a_cam_pos: [f32; 3],
    _pad: [f32; 1],
}

const RESOLUTION: usize = 512;
const HALF_RESOLUTION: usize = 128;
const DOMAIN_SIZE: f32 = 1000.0;

const COLOR_RANGE: i::SubresourceRange = i::SubresourceRange {
    aspects: f::Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

fn translate_shader(code: &str, stage: pso::Stage) -> Result<Vec<u32>, String> {
    use glsl_to_spirv::{compile, ShaderType};

    let ty = match stage {
        pso::Stage::Vertex => ShaderType::Vertex,
        pso::Stage::Fragment => ShaderType::Fragment,
        pso::Stage::Geometry => ShaderType::Geometry,
        pso::Stage::Hull => ShaderType::TessellationControl,
        pso::Stage::Domain => ShaderType::TessellationEvaluation,
        pso::Stage::Compute => ShaderType::Compute,
    };

    compile(code, ty).map(|out| pso::read_spirv(out).unwrap())
}

#[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
fn main() -> Result<(), ()> {
    unsafe {
    env_logger::init().unwrap();

    let mut events_loop = winit::EventsLoop::new();
    let wb = winit::WindowBuilder::new()
        .with_dimensions((1200, 800).into())
        .with_title("ocean".to_string());
    let window = wb.build(&events_loop).unwrap();

    let PhysicalSize {
        width: pixel_width,
        height: pixel_height,
    } = window.get_inner_size().unwrap().to_physical(window.get_hidpi_factor());

    let mut camera =
        camera::Camera::new(glm::vec3(-110.0, 150.0, 200.0), glm::vec3(-1.28, -0.44, 0.0));

    let perspective: [[f32; 4]; 4] = {
        let aspect_ratio = pixel_width as f32 / pixel_height as f32;
        glm::perspective(
            aspect_ratio,
            glm::half_pi::<f32>() * 0.8,
            0.1,
            1024.0,
        ).into()
    };

    let (_instance, mut adapters, mut surface) = {
        let instance = back::Instance::create("gfx-ocean", 1);
        let surface = instance.create_surface(&window);
        let adapters = instance.enumerate_adapters();
        (instance, adapters, surface)
    };

    let mut adapter = adapters.remove(0);
    let (caps, formats, _) = surface
        .compatibility(&mut adapter.physical_device);
    let surface_format = formats
        .map_or(f::Format::Rgba8Srgb, |formats| {
            formats
                .into_iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .unwrap()
        });
    let memory_types = adapter.physical_device.memory_properties().memory_types;

    let (mut device, mut queue_group) = adapter
        .open_with::<_, hal::General>(1, |family| surface.supports_queue_family(family))
        .unwrap();

    let mut general_pool =
        device.create_command_pool_typed(&queue_group, pool::CommandPoolCreateFlags::empty())
        .map_err(|_| ())?;

    let mut swap_config = SwapchainConfig::from_caps(&caps, surface_format, Extent2D {
        width: pixel_width as _,
        height: pixel_height as _
    });
    swap_config.present_mode = hal::window::PresentMode::Immediate; // disable vsync

    let (mut swap_chain, swap_images) = device.create_swapchain(&mut surface, swap_config, None)
        .map_err(|_| ())?;
    let frame_images = swap_images
            .into_iter()
            .map(|image| {
                let rtv = device
                    .create_image_view(
                        &image,
                        i::ViewKind::D2,
                        surface_format,
                        Swizzle::NO,
                        COLOR_RANGE,
                    )
                    .unwrap();
                (image, rtv)
            })
            .collect::<Vec<_>>();

    let frames_in_flight = 3;

    let mut upload_fence = device.create_fence(false)
        .map_err(|_| ())?;

    let mut image_acquire_semaphores = Vec::with_capacity(frame_images.len());
    let mut free_acquire_semaphore = device
        .create_semaphore()
        .map_err(|_| ())?;

    let mut submission_complete_semaphores = Vec::with_capacity(frames_in_flight);
    let mut submission_complete_fences = Vec::with_capacity(frames_in_flight);

    let mut cmd_pools = Vec::with_capacity(frames_in_flight);
    let mut cmd_buffers = Vec::with_capacity(frames_in_flight);

    for _ in 0..frames_in_flight {
        cmd_pools.push(
            device
                .create_command_pool_typed(&queue_group, pool::CommandPoolCreateFlags::empty())
                .map_err(|_| ())?
        );
    }

    for _ in 0..frame_images.len() {
        image_acquire_semaphores.push(
            device
                .create_semaphore()
                .map_err(|_| ())?
        );
    }

    for i in 0..frames_in_flight {
        submission_complete_semaphores.push(
            device
                .create_semaphore()
                .map_err(|_| ())?
        );
        submission_complete_fences.push(
            device
                .create_fence(true)
                .map_err(|_| ())?
        );
        cmd_buffers.push(cmd_pools[i].acquire_command_buffer::<command::OneShot>());
    }


    let depth_format = f::Format::D32Sfloat;
    let mut depth_image = device
        .create_image(
            i::Kind::D2(pixel_width as _, pixel_height as _, 1, 1),
            1,
            depth_format,
            i::Tiling::Optimal,
            i::Usage::DEPTH_STENCIL_ATTACHMENT,
            i::ViewCapabilities::empty(),
        )
        .unwrap();

    let depth_mem_reqs = device.get_image_requirements(&depth_image);

    let mem_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, mem_type)| {
            depth_mem_reqs.type_mask & (1 << id) != 0
                && mem_type.properties.contains(m::Properties::DEVICE_LOCAL)
        })
        .unwrap()
        .into();

    let depth_memory = device
        .allocate_memory(mem_type, depth_mem_reqs.size)
        .unwrap();
    device
        .bind_image_memory(&depth_memory, 0, &mut depth_image)
        .map_err(|_| ())?;

    let depth_view = device
        .create_image_view(
            &depth_image,
            i::ViewKind::D2,
            depth_format,
            f::Swizzle::NO,
            i::SubresourceRange {
                aspects: f::Aspects::DEPTH,
                levels: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();

    let vs_ocean = {
        let mut file = fs::File::open("shader/ocean.vert").unwrap();
        let mut shader = String::new();
        file.read_to_string(&mut shader);

        device
            .create_shader_module(&translate_shader(&shader, pso::Stage::Vertex).unwrap())
            .unwrap()
    };

    let fs_ocean = {
        let mut file = fs::File::open("shader/ocean.frag").unwrap();
        let mut shader = String::new();
        file.read_to_string(&mut shader);

        device
            .create_shader_module(&translate_shader(&shader, pso::Stage::Fragment).unwrap())
            .unwrap()
    };

    let fft = fft::Fft::init(&mut device)
        .map_err(|_| ())?;
    let propagate = ocean::Propagation::init(&mut device)
        .map_err(|_| ())?;
    let correction = ocean::Correction::init(&mut device)
        .map_err(|_| ())?;

    let set_layout = device.create_descriptor_set_layout(&[
        pso::DescriptorSetLayoutBinding {
            binding: 0,
            ty: pso::DescriptorType::UniformBuffer,
            count: 1,
            stage_flags: pso::ShaderStageFlags::VERTEX | pso::ShaderStageFlags::FRAGMENT, immutable_samplers: false
        },
        pso::DescriptorSetLayoutBinding {
            binding: 1,
            ty: pso::DescriptorType::SampledImage,
            count: 1,
            stage_flags: pso::ShaderStageFlags::VERTEX | pso::ShaderStageFlags::FRAGMENT, immutable_samplers: false
        },
        pso::DescriptorSetLayoutBinding {
            binding: 2,
            ty: pso::DescriptorType::Sampler,
            count: 1,
            stage_flags: pso::ShaderStageFlags::VERTEX | pso::ShaderStageFlags::FRAGMENT, immutable_samplers: false
        },
    ], &[])
    .map_err(|_| ())?;

    let ocean_layout = device.create_pipeline_layout(Some(&set_layout), &[])
        .map_err(|_| ())?;
    let ocean_pass = {
        let attachment = pass::Attachment {
            format: Some(surface_format),
            samples: 1,
            ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::Clear,
                pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: pass::AttachmentOps::DONT_CARE,
            layouts: i::Layout::Undefined..i::Layout::Present,
        };

        let depth_attachment = pass::Attachment {
            format: Some(depth_format),
            samples: 1,
            ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::Clear,
                pass::AttachmentStoreOp::DontCare,
            ),
            stencil_ops: pass::AttachmentOps::DONT_CARE,
            layouts: i::Layout::Undefined..i::Layout::DepthStencilAttachmentOptimal,
        };

        let subpass = pass::SubpassDesc {
            colors: &[(0, i::Layout::ColorAttachmentOptimal)],
            depth_stencil: Some(&(1, i::Layout::DepthStencilAttachmentOptimal)),
            inputs: &[],
            preserves: &[],
            resolves: &[],
        };

        device.create_render_pass(&[attachment, depth_attachment], &[subpass], &[])
            .map_err(|_| ())?
    };

    let extent = i::Extent {
        width: pixel_width as _,
        height: pixel_height as _,
        depth: 1,
    };
    let ocean_framebuffers = frame_images
        .iter()
        .map(|&(_, ref rtv)| {
            device
                .create_framebuffer(&ocean_pass, vec![rtv, &depth_view], extent)
                .unwrap()
        })
        .collect::<Vec<_>>();

    let pipelines = {
        let (vs_entry, fs_entry) = (
            pso::EntryPoint {
                entry: "main",
                module: &vs_ocean,
                specialization: pso::Specialization::default(),
            },
            pso::EntryPoint {
                entry: "main",
                module: &fs_ocean,
                specialization: pso::Specialization::default(),
            },
        );

        let shader_entries = pso::GraphicsShaderSet {
            vertex: vs_entry,
            hull: None,
            domain: None,
            geometry: None,
            fragment: Some(fs_entry),
        };

        let subpass = pass::Subpass {
            index: 0,
            main_pass: &ocean_pass,
        };

        let mut ocean_pipe_desc = pso::GraphicsPipelineDesc::new(
            shader_entries,
            Primitive::TriangleList,
            pso::Rasterizer {
                polygon_mode: pso::PolygonMode::Fill,
                cull_face: pso::Face::NONE,
                front_face: pso::FrontFace::CounterClockwise,
                depth_clamping: false,
                depth_bias: None,
                conservative: false,
            },
            &ocean_layout,
            subpass,
        );

        ocean_pipe_desc.depth_stencil = pso::DepthStencilDesc {
            depth: Some(pso::DepthTest {
                fun: pso::Comparison::LessEqual,
                write: true,
            }),
            depth_bounds: false,
            stencil: None,
        };
        ocean_pipe_desc.rasterizer.depth_clamping = false;
        ocean_pipe_desc.blender.targets.push(pso::ColorBlendDesc {
            mask: pso::ColorMask::ALL,
            blend: None,
        });
        ocean_pipe_desc.vertex_buffers.push(pso::VertexBufferDesc {
            binding: 0,
            stride: std::mem::size_of::<Vertex>() as u32,
            rate: pso::VertexInputRate::Vertex,
        });
        ocean_pipe_desc.vertex_buffers.push(pso::VertexBufferDesc {
            binding: 1,
            stride: std::mem::size_of::<PatchOffset>() as u32,
            rate: pso::VertexInputRate::Instance(1),
        });
        ocean_pipe_desc.attributes.push(pso::AttributeDesc {
            location: 0,
            binding: 0,
            element: pso::Element {
                format: Format::Rgb32Sfloat,
                offset: 0,
            },
        });
        ocean_pipe_desc.attributes.push(pso::AttributeDesc {
            location: 1,
            binding: 0,
            element: pso::Element {
                format: Format::Rg32Sfloat,
                offset: 12,
            },
        });
        ocean_pipe_desc.attributes.push(pso::AttributeDesc {
            location: 2,
            binding: 1,
            element: pso::Element {
                format: Format::Rg32Sfloat,
                offset: 0,
            },
        });

        device.create_graphics_pipelines(&[ocean_pipe_desc], None)
    };

    let sampler = device.create_sampler(i::SamplerInfo::new(i::Filter::Linear, i::WrapMode::Tile))
        .map_err(|_| ())?;

    let mut desc_pool = device.create_descriptor_pool(
        1, // sets
        &[
            pso::DescriptorRangeDesc {
                ty: pso::DescriptorType::UniformBuffer,
                count: 1,
            },
            pso::DescriptorRangeDesc {
                ty: pso::DescriptorType::SampledImage,
                count: 1,
            },
            pso::DescriptorRangeDesc {
                ty: pso::DescriptorType::Sampler,
                count: 1,
            },
        ],
        pso::DescriptorPoolCreateFlags::empty()
    )
    .map_err(|_| ())?;

    let desc_set = desc_pool.allocate_set(&set_layout)
        .map_err(|_| ())?;

    let (locals_buffer, locals_memory) = {
        let buffer_stride = std::mem::size_of::<Locals>() as u64;
        let buffer_len = buffer_stride;
        let mut locals_buffer = device.create_buffer(buffer_len, b::Usage::UNIFORM).unwrap();
        let buffer_req = device.get_buffer_requirements(&locals_buffer);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
        device.bind_buffer_memory(&buffer_memory, 0, &mut locals_buffer)
            .map_err(|_| ())?;

        {
            let mut locals = device
                .acquire_mapping_writer::<Locals>(&buffer_memory, 0..buffer_len)
                .unwrap();
            locals[0] = Locals {
                a_proj: perspective.into(),
                a_view: camera.view().into(),
                a_cam_pos: camera.position().into(),
                _pad: [0.0; 1],
            };
            device.release_mapping_writer(locals)
                .map_err(|_| ())?;
        }

        (locals_buffer, buffer_memory)
    };

    // grid
    let (grid_vertex_buffer, grid_vertex_memory) = {
        let buffer_stride = std::mem::size_of::<Vertex>() as u64;
        let buffer_len = (HALF_RESOLUTION * HALF_RESOLUTION) as u64 * buffer_stride;
        let mut vertex_buffer = device.create_buffer(buffer_len, b::Usage::VERTEX).unwrap();
        let buffer_req = device.get_buffer_requirements(&vertex_buffer);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
        device
            .bind_buffer_memory(&buffer_memory, 0, &mut vertex_buffer)
            .map_err(|_| ())?;

        {
            let mut vertices = device
                .acquire_mapping_writer::<Vertex>(&buffer_memory, 0..buffer_len)
                .unwrap();
            for z in 0..HALF_RESOLUTION {
                for x in 0..HALF_RESOLUTION {
                    vertices[z * HALF_RESOLUTION + x] = Vertex {
                        a_Pos: [x as f32, 0.0f32, z as f32],
                        a_Uv: [
                            (x as f32) / (HALF_RESOLUTION - 1) as f32,
                            (z as f32) / (HALF_RESOLUTION - 1) as f32,
                        ],
                    };
                }
            }
            device.release_mapping_writer(vertices)
                .map_err(|_| ())?;
        }

        (vertex_buffer, buffer_memory)
    };

    let (grid_patch_buffer, grid_patch_memory) = {
        let buffer_stride = std::mem::size_of::<PatchOffset>() as u64;
        let buffer_len = 4 * buffer_stride;
        let mut buffer = device.create_buffer(buffer_len, b::Usage::VERTEX).unwrap();
        let buffer_req = device.get_buffer_requirements(&buffer);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
        device
            .bind_buffer_memory(&buffer_memory, 0, &mut buffer)
            .map_err(|_| ())?;

        {
            let mut patch = device
                .acquire_mapping_writer::<PatchOffset>(&buffer_memory, 0..buffer_len)
                .unwrap();
            patch[0] = PatchOffset {
                a_offset: [0.0, 0.0],
            };
            patch[1] = PatchOffset {
                a_offset: [(HALF_RESOLUTION - 1) as f32, 0.0],
            };
            patch[2] = PatchOffset {
                a_offset: [0.0, (HALF_RESOLUTION - 1) as f32],
            };
            patch[3] = PatchOffset {
                a_offset: [(HALF_RESOLUTION - 1) as f32, (HALF_RESOLUTION - 1) as f32],
            };
            device.release_mapping_writer(patch)
                .map_err(|_| ())?;
        }

        (buffer, buffer_memory)
    };

    let (grid_index_buffer, grid_index_memory) = {
        let buffer_stride = std::mem::size_of::<u32>() as u64;
        let buffer_len = (6 * (HALF_RESOLUTION - 1) * (HALF_RESOLUTION - 1)) as u64 * buffer_stride;
        let mut index_buffer = device.create_buffer(buffer_len, b::Usage::INDEX).unwrap();
        let buffer_req = device.get_buffer_requirements(&index_buffer);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
        device
            .bind_buffer_memory(&buffer_memory, 0, &mut index_buffer)
            .map_err(|_| ())?;

        {
            let mut indices = device
                .acquire_mapping_writer::<u32>(&buffer_memory, 0..buffer_len)
                .unwrap();
            for z in 0..HALF_RESOLUTION - 1 {
                for x in 0..HALF_RESOLUTION - 1 {
                    let i = z * (HALF_RESOLUTION - 1) + x;
                    indices[6 * i] = (z * HALF_RESOLUTION + x) as _;
                    indices[6 * i + 1] = ((z + 1) * HALF_RESOLUTION + x) as _;
                    indices[6 * i + 2] = (z * HALF_RESOLUTION + x + 1) as _;
                    indices[6 * i + 3] = (z * HALF_RESOLUTION + x + 1) as _;
                    indices[6 * i + 4] = ((z + 1) * HALF_RESOLUTION + x) as _;
                    indices[6 * i + 5] = ((z + 1) * HALF_RESOLUTION + x + 1) as _;
                }
            }
            device.release_mapping_writer(indices);
        }

        (index_buffer, buffer_memory)
    };

    // ocean data
    let (initial_spec, dx_spec, dy_spec, dz_spec, spectrum_memory) = {
        let buffer_stride = 2 * std::mem::size_of::<f32>() as u64;
        let buffer_len = (RESOLUTION * RESOLUTION) as u64 * buffer_stride;

        let mut initial_spec = device
            .create_buffer(buffer_len, b::Usage::STORAGE | b::Usage::TRANSFER_DST)
            .unwrap();
        let mut dx_spec = device.create_buffer(buffer_len, b::Usage::STORAGE).unwrap();
        let mut dy_spec = device.create_buffer(buffer_len, b::Usage::STORAGE).unwrap();
        let mut dz_spec = device.create_buffer(buffer_len, b::Usage::STORAGE).unwrap();

        let buffer_req = device.get_buffer_requirements(&initial_spec);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let buffer_memory = device
            .allocate_memory(mem_type, 16 * buffer_req.size)
            .unwrap();
        device
            .bind_buffer_memory(&buffer_memory, 0, &mut initial_spec)
            .unwrap();

        let dx_offset = align_up(buffer_len, buffer_req.alignment);
        let dy_offset = align_up(dx_offset + buffer_len, buffer_req.alignment);
        let dz_offset = align_up(dy_offset + buffer_len, buffer_req.alignment);
        device
            .bind_buffer_memory(&buffer_memory, dx_offset, &mut dx_spec);
        device
            .bind_buffer_memory(&buffer_memory, dy_offset, &mut dy_spec);
        device
            .bind_buffer_memory(&buffer_memory, dz_offset, &mut dz_spec);

        (initial_spec, dx_spec, dy_spec, dz_spec, buffer_memory)
    };

    let (omega_buffer, omega_memory) = {
        let buffer_stride = std::mem::size_of::<f32>() as u64;
        let buffer_len = (RESOLUTION * RESOLUTION) as u64 * buffer_stride;
        let mut omega_buffer = device
            .create_buffer(buffer_len, b::Usage::STORAGE | b::Usage::TRANSFER_DST)
            .unwrap();
        let buffer_req = device.get_buffer_requirements(&omega_buffer);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
        device
            .bind_buffer_memory(&buffer_memory, 0, &mut omega_buffer);

        (omega_buffer, buffer_memory)
    };

    let (propagate_locals_buffer, propagate_locals_memory) = {
        let buffer_stride = std::mem::size_of::<PropagateLocals>() as u64;
        let buffer_len = buffer_stride;
        let mut locals_buffer = device.create_buffer(buffer_len, b::Usage::UNIFORM).unwrap();
        let buffer_req = device.get_buffer_requirements(&locals_buffer);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
        device
            .bind_buffer_memory(&buffer_memory, 0, &mut locals_buffer);
        (locals_buffer, buffer_memory)
    };

    let (correct_locals_buffer, correct_locals_memory) = {
        let buffer_stride = std::mem::size_of::<CorrectionLocals>() as u64;
        let buffer_len = buffer_stride;
        let mut locals_buffer = device.create_buffer(buffer_len, b::Usage::UNIFORM).unwrap();
        let buffer_req = device.get_buffer_requirements(&locals_buffer);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
        device
            .bind_buffer_memory(&buffer_memory, 0, &mut locals_buffer);

        {
            let mut locals = device
                .acquire_mapping_writer::<CorrectionLocals>(&buffer_memory, 0..buffer_len)
                .unwrap();
            locals[0] = CorrectionLocals {
                resolution: RESOLUTION as _,
            };
            device.release_mapping_writer(locals);
        }

        (locals_buffer, buffer_memory)
    };

    let viewport = pso::Viewport {
        rect: pso::Rect {
            x: 0,
            y: 0,
            w: pixel_width as _,
            h: pixel_height as _,
        },
        depth: 0.0..1.0,
    };
    let scissor = pso::Rect {
        x: 0,
        y: 0,
        w: pixel_width as _,
        h: pixel_height as _,
    };

    let spectrum_len = RESOLUTION * RESOLUTION * 2 * std::mem::size_of::<f32>();
    let omega_len = RESOLUTION * RESOLUTION * std::mem::size_of::<f32>();

    // Upload initial data
    let (omega_staging_buffer, omega_staging_memory) = {
        let buffer_stride = std::mem::size_of::<f32>() as u64;
        let buffer_len = (RESOLUTION * RESOLUTION) as u64 * buffer_stride;
        let mut staging_buffer = device
            .create_buffer(buffer_len, b::Usage::TRANSFER_SRC)
            .unwrap();
        let buffer_req = device.get_buffer_requirements(&staging_buffer);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
        device
            .bind_buffer_memory(&buffer_memory, 0, &mut staging_buffer);

        {
            let mut data = device
                .acquire_mapping_writer::<f32>(&buffer_memory, 0..buffer_len)
                .unwrap();
            let mut file = fs::File::open("data/omega.bin").unwrap();
            let mut contents = Vec::new();
            file.read_to_end(&mut contents);
            let omega: Vec<f32> = bincode::deserialize(&contents[..]).unwrap();
            data.copy_from_slice(&omega);
            device.release_mapping_writer(data);
        }

        (staging_buffer, buffer_memory)
    };

    let (spec_staging_buffer, spec_staging_memory) = {
        let buffer_stride = 2 * std::mem::size_of::<f32>() as u64;
        let buffer_len = (RESOLUTION * RESOLUTION) as u64 * buffer_stride;
        let mut staging_buffer = device
            .create_buffer(buffer_len, b::Usage::TRANSFER_SRC)
            .unwrap();
        let buffer_req = device.get_buffer_requirements(&staging_buffer);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        let buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
        device
            .bind_buffer_memory(&buffer_memory, 0, &mut staging_buffer);

        {
            let mut data = device
                .acquire_mapping_writer::<[f32; 2]>(&buffer_memory, 0..buffer_len)
                .unwrap();

            let mut file = fs::File::open("data/spectrum.bin").unwrap();
            let mut contents = Vec::new();
            file.read_to_end(&mut contents);
            let spectrum: Vec<[f32; 2]> = bincode::deserialize(&contents[..]).unwrap();
            data.copy_from_slice(&spectrum);
            device.release_mapping_writer(data);
        }

        (staging_buffer, buffer_memory)
    };

    let kind = i::Kind::D2(RESOLUTION as i::Size, RESOLUTION as i::Size, 1, 1);
    let img_format = Format::Rgba32Sfloat;
    let mut displacement_map = device
        .create_image(
            kind,
            1,
            img_format,
            i::Tiling::Optimal,
            i::Usage::SAMPLED | i::Usage::STORAGE,
            i::ViewCapabilities::empty(),
        )
        .unwrap(); // TODO: usage
    let image_req = device.get_image_requirements(&displacement_map);

    let device_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, mem_type)| {
            image_req.type_mask & (1 << id) != 0
                && mem_type.properties.contains(m::Properties::DEVICE_LOCAL)
        })
        .unwrap()
        .into();

    let image_memory = device.allocate_memory(device_type, image_req.size).unwrap();
    device
        .bind_image_memory(&image_memory, 0, &mut displacement_map)
        .map_err(|_| ())?;
    let displacement_uav = device
        .create_image_view(
            &displacement_map,
            i::ViewKind::D2,
            img_format,
            Swizzle::NO,
            COLOR_RANGE,
        )
        .unwrap();
    let displacement_srv = device
        .create_image_view(
            &displacement_map,
            i::ViewKind::D2,
            img_format,
            Swizzle::NO,
            COLOR_RANGE,
        )
        .unwrap();

    // Upload data
    {
        let mut cmd_buffer = general_pool.acquire_command_buffer::<command::OneShot>();
        cmd_buffer.begin();

        let image_barrier = m::Barrier::Image {
            states: (i::Access::empty(), i::Layout::Undefined)
                ..(
                    i::Access::SHADER_READ | i::Access::SHADER_WRITE,
                    i::Layout::General,
                ),
            target: &displacement_map,
            families: None,
            range: COLOR_RANGE,
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::COMPUTE_SHADER,
            m::Dependencies::empty(),
            &[image_barrier],
        );

        // TODO: pipeline barriers
        cmd_buffer.copy_buffer(
            &spec_staging_buffer,
            &initial_spec,
            &[command::BufferCopy {
                src: 0,
                dst: 0,
                size: (RESOLUTION * RESOLUTION * 2 * std::mem::size_of::<f32>()) as u64,
            }],
        );

        cmd_buffer.copy_buffer(
            &omega_staging_buffer,
            &omega_buffer,
            &[command::BufferCopy {
                src: 0,
                dst: 0,
                size: (RESOLUTION * RESOLUTION * std::mem::size_of::<f32>()) as u64,
            }],
        );
        cmd_buffer.finish();

        queue_group.queues[0].submit_without_semaphores(Some(&cmd_buffer), Some(&mut upload_fence));
        device.wait_for_fence(&upload_fence, !0);
    }

    device.write_descriptor_sets(vec![
        pso::DescriptorSetWrite {
            set: &desc_set,
            binding: 0,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(&locals_buffer, None..None)),
        },
        pso::DescriptorSetWrite {
            set: &desc_set,
            binding: 1,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Image(
                &displacement_srv,
                i::Layout::General,
            )),
        },
        pso::DescriptorSetWrite {
            set: &desc_set,
            binding: 2,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Sampler(&sampler)),
        },
        pso::DescriptorSetWrite {
            set: &propagate.desc_set,
            binding: 0,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &propagate_locals_buffer,
                None..None,
            )),
        },
        pso::DescriptorSetWrite {
            set: &propagate.desc_set,
            binding: 1,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &initial_spec,
                None..Some(spectrum_len as u64),
            )),
        },
        pso::DescriptorSetWrite {
            set: &propagate.desc_set,
            binding: 2,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &omega_buffer,
                None..Some(omega_len as u64),
            )),
        },
        pso::DescriptorSetWrite {
            set: &propagate.desc_set,
            binding: 3,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &dy_spec,
                None..Some(spectrum_len as u64),
            )),
        },
        pso::DescriptorSetWrite {
            set: &propagate.desc_set,
            binding: 4,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &dx_spec,
                None..Some(spectrum_len as u64),
            )),
        },
        pso::DescriptorSetWrite {
            set: &propagate.desc_set,
            binding: 5,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &dz_spec,
                None..Some(spectrum_len as u64),
            )),
        },
    ]);

    device.write_descriptor_sets(vec![
        pso::DescriptorSetWrite {
            set: &correction.desc_set,
            binding: 0,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(&correct_locals_buffer, None..None)),
        },
        pso::DescriptorSetWrite {
            set: &correction.desc_set,
            binding: 1,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &dy_spec,
                None..Some(spectrum_len as u64),
            )),
        },
        pso::DescriptorSetWrite {
            set: &correction.desc_set,
            binding: 2,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &dx_spec,
                None..Some(spectrum_len as u64),
            )),
        },
        pso::DescriptorSetWrite {
            set: &correction.desc_set,
            binding: 3,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &dz_spec,
                None..Some(spectrum_len as u64),
            )),
        },
        pso::DescriptorSetWrite {
            set: &correction.desc_set,
            binding: 4,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Image(
                &displacement_uav,
                i::Layout::General,
            )),
        },
    ]);

    device.write_descriptor_sets(vec![
        pso::DescriptorSetWrite {
            set: &fft.desc_sets[0],
            binding: 0,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &dx_spec,
                None..Some(spectrum_len as u64),
            )),
        },
        pso::DescriptorSetWrite {
            set: &fft.desc_sets[1],
            binding: 0,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &dy_spec,
                None..Some(spectrum_len as u64),
            )),
        },
        pso::DescriptorSetWrite {
            set: &fft.desc_sets[2],
            binding: 0,
            array_offset: 0,
            descriptors: Some(pso::Descriptor::Buffer(
                &dz_spec,
                None..Some(spectrum_len as u64),
            )),
        },
    ]);

    let time_start = Instant::now();
    let mut time_last = time_start;

    let mut running = true;
    let mut frame_id = 0;
    let mut avg_cpu_time = 0.0;

    while running {
        events_loop.poll_events(|event| match event {
            winit::Event::WindowEvent { event, .. } => match event {
                winit::WindowEvent::KeyboardInput {
                    input:
                        winit::KeyboardInput {
                            virtual_keycode: Some(winit::VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                }
                | winit::WindowEvent::CloseRequested => running = false,
                winit::WindowEvent::KeyboardInput { input, .. } => {
                    camera.handle_event(input);
                }
                _ => (),
            },
            _ => (),
        });

        let time_now = Instant::now();
        let elapsed = time_now.duration_since(time_last).as_micros() as f32 / 1_000_000.0;
        let current = time_now.duration_since(time_start).as_micros() as f32 / 1_000_000.0;
        time_last = time_now;

        let factor = 0.1;
        avg_cpu_time = avg_cpu_time * (1.0 - factor) + elapsed * factor;
        window.set_title(&format!("gfx-ocean :: {:.*} ms", 2, avg_cpu_time * 1000.0));

        let (swap_image, _) = swap_chain.acquire_image(!0, Some(&free_acquire_semaphore), None)
            .map_err(|_| ())?;

        core::mem::swap(
            &mut free_acquire_semaphore,
            &mut image_acquire_semaphores[swap_image as usize],
        );

        let frame_idx = frame_id as usize % frames_in_flight;

            device
                .wait_for_fence(&submission_complete_fences[frame_idx], !0)
                .map_err(|_| ())?;
            device
                .reset_fence(&submission_complete_fences[frame_idx])
                .map_err(|_| ())?;
            cmd_pools[frame_idx].reset(false);

        // Rendering
        let cmd_buffer = &mut cmd_buffers[frame_idx];

        // Update view
        camera.update(elapsed);
        let mut locals = device
            .acquire_mapping_writer::<Locals>(
                &locals_memory,
                0..std::mem::size_of::<Locals>() as u64,
            )
            .unwrap();
        locals[0] = Locals {
            a_proj: perspective.into(),
            a_view: camera.view().into(),
            a_cam_pos: camera.position().into(),
            _pad: [0.0; 1],
        };
        device.release_mapping_writer(locals)
            .map_err(|_| ())?;

        let mut locals = device
            .acquire_mapping_writer::<PropagateLocals>(
                &propagate_locals_memory,
                0..std::mem::size_of::<PropagateLocals>() as u64,
            )
            .unwrap();
        locals[0] = PropagateLocals {
            time: current,
            resolution: RESOLUTION as i32,
            domain_size: DOMAIN_SIZE,
        };
        device.release_mapping_writer(locals)
            .map_err(|_| ())?;

        cmd_buffer.begin();
        cmd_buffer.bind_compute_pipeline(&propagate.pipeline);
        cmd_buffer.bind_compute_descriptor_sets(
            &propagate.layout,
            0,
            Some(&propagate.desc_set),
            &[]
        );
        cmd_buffer.dispatch([RESOLUTION as u32, RESOLUTION as u32, 1]);

        let dx_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &dx_spec,
            families: None,
            range: None..None,
        };
        let dy_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &dy_spec,
            families: None,
            range: None..None,
        };
        let dz_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &dz_spec,
            families: None,
            range: None..None,
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::COMPUTE_SHADER..pso::PipelineStage::COMPUTE_SHADER,
            m::Dependencies::empty(),
            &[dx_barrier, dy_barrier, dz_barrier],
        );

        cmd_buffer.bind_compute_pipeline(&fft.row_pass);
        cmd_buffer.bind_compute_descriptor_sets(&fft.layout, 0, &fft.desc_sets[0..1], &[]);
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);
        cmd_buffer.bind_compute_descriptor_sets(&fft.layout, 0, &fft.desc_sets[1..2], &[]);
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);
        cmd_buffer.bind_compute_descriptor_sets(&fft.layout, 0, &fft.desc_sets[2..3], &[]);
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);

        let dx_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &dx_spec,
            families: None,
            range: None..None,
        };
        let dy_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &dy_spec,
            families: None,
            range: None..None,
        };
        let dz_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &dz_spec,
            families: None,
            range: None..None,
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::COMPUTE_SHADER..pso::PipelineStage::COMPUTE_SHADER,
            m::Dependencies::empty(),
            &[dx_barrier, dy_barrier, dz_barrier],
        );

        cmd_buffer.bind_compute_pipeline(&fft.col_pass);
        cmd_buffer.bind_compute_descriptor_sets(&fft.layout, 0, &fft.desc_sets[0..1], &[]);
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);
        cmd_buffer.bind_compute_descriptor_sets(&fft.layout, 0, &fft.desc_sets[1..2], &[]);
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);
        cmd_buffer.bind_compute_descriptor_sets(&fft.layout, 0, &fft.desc_sets[2..3], &[]);
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);

        let dx_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &dx_spec,
            families: None,
            range: None..None,
        };
        let dy_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &dy_spec,
            families: None,
            range: None..None,
        };
        let dz_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &dz_spec,
            families: None,
            range: None..None,
        };
        let image_barrier = m::Barrier::Image {
            states: (
                i::Access::SHADER_READ | i::Access::SHADER_WRITE,
                i::Layout::General,
            )
                ..(
                    i::Access::SHADER_READ | i::Access::SHADER_WRITE,
                    i::Layout::General,
                ),
            target: &displacement_map,
            families: None,
            range: COLOR_RANGE,
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::VERTEX_SHADER | pso::PipelineStage::COMPUTE_SHADER
                ..pso::PipelineStage::COMPUTE_SHADER,
            m::Dependencies::empty(),
            &[dx_barrier, dy_barrier, dz_barrier, image_barrier],
        );

        cmd_buffer.bind_compute_pipeline(&correction.pipeline);
        cmd_buffer.bind_compute_descriptor_sets(
            &correction.layout,
            0,
            Some(&correction.desc_set), &[]
        );
        cmd_buffer.dispatch([RESOLUTION as u32, RESOLUTION as u32, 1]);

        let image_barrier = m::Barrier::Image {
            states: (
                i::Access::SHADER_READ | i::Access::SHADER_WRITE,
                i::Layout::General,
            )
                ..(
                    i::Access::SHADER_READ | i::Access::SHADER_WRITE,
                    i::Layout::General,
                ),
            target: &displacement_map,
            families: None,
            range: COLOR_RANGE,
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::COMPUTE_SHADER
                ..pso::PipelineStage::VERTEX_SHADER | pso::PipelineStage::FRAGMENT_SHADER,
            m::Dependencies::empty(),
            &[image_barrier],
        );

        cmd_buffer.set_viewports(0, &[viewport.clone()]);
        cmd_buffer.set_scissors(0, &[scissor]);
        cmd_buffer.bind_graphics_pipeline(&pipelines[0].as_ref().unwrap());
        cmd_buffer.bind_graphics_descriptor_sets(&ocean_layout, 0, Some(&desc_set), &[]);
        cmd_buffer.bind_vertex_buffers(0, vec![
            (&grid_vertex_buffer, 0),
            (&grid_patch_buffer, 0),
        ]);
        cmd_buffer.bind_index_buffer(b::IndexBufferView {
            buffer: &grid_index_buffer,
            offset: 0,
            index_type: IndexType::U32,
        });

        {
            let mut encoder = cmd_buffer.begin_render_pass_inline(
                &ocean_pass,
                &ocean_framebuffers[swap_image as usize],
                pso::Rect {
                    x: 0,
                    y: 0,
                    w: pixel_width as _,
                    h: pixel_height as _,
                },
                &[
                    ClearValue::Color(ClearColor::Sfloat([0.6, 0.6, 0.6, 1.0])),
                    ClearValue::DepthStencil(ClearDepthStencil(1.0, 0)),
                ],
            );
            let num_indices = 6 * (HALF_RESOLUTION - 1) * (HALF_RESOLUTION - 1);
            encoder.draw_indexed(0..num_indices as u32, 0, 0..4);
        }

        cmd_buffer.finish();

        let submission = Submission {
            command_buffers: Some(&*cmd_buffer),
            wait_semaphores: vec![(&image_acquire_semaphores[swap_image as usize], pso::PipelineStage::BOTTOM_OF_PIPE)],
            signal_semaphores: Some(&submission_complete_semaphores[frame_idx]),
        };
        queue_group.queues[0].submit(submission, Some(&submission_complete_fences[frame_idx]));

        swap_chain.present(&mut queue_group.queues[0], swap_image, Some(&submission_complete_semaphores[frame_idx]))
            .map_err(|_| ())?;

        frame_id += 1;
    }

    // cleanup
    fft.destroy(&mut device);
    propagate.destroy(&mut device);
    correction.destroy(&mut device);

    device.destroy_descriptor_pool(desc_pool);
    device.destroy_descriptor_set_layout(set_layout);

    device.destroy_shader_module(vs_ocean);
    device.destroy_shader_module(fs_ocean);

    device.destroy_buffer(grid_index_buffer);
    device.destroy_buffer(grid_vertex_buffer);
    device.destroy_buffer(locals_buffer);
    device.destroy_buffer(initial_spec);
    device.destroy_buffer(dx_spec);
    device.destroy_buffer(dy_spec);
    device.destroy_buffer(dz_spec);
    device.destroy_buffer(propagate_locals_buffer);

    device.free_memory(grid_index_memory);
    device.free_memory(grid_vertex_memory);
    device.free_memory(grid_patch_memory);
    device.free_memory(omega_memory);
    device.free_memory(locals_memory);
    device.free_memory(spectrum_memory);
    device.free_memory(propagate_locals_memory);
    device.free_memory(correct_locals_memory);
    device.free_memory(omega_staging_memory);
    device.free_memory(spec_staging_memory);

    for pipeline in pipelines {
        if let Ok(pipeline) = pipeline {
            device.destroy_graphics_pipeline(pipeline);
        }
    }

    for framebuffer in ocean_framebuffers {
        device.destroy_framebuffer(framebuffer);
    }

    for (_, rtv) in frame_images {
        device.destroy_image_view(rtv);
    }

    Ok(())
}}

fn align_up(value: u64, alignment: u64) -> u64 {
    ((value + alignment - 1) / alignment) * alignment
}

#[cfg(not(any(feature = "vulkan", feature = "dx12", feature = "metal")))]
fn main() {
    println!("You need to enable the one of the following API backends: vulkan, dx12 or metal");
}
