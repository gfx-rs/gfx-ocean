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

use std::{borrow::Borrow, fs, io::Read, iter, time::Instant};

use gfx_hal::{
    adapter::PhysicalDevice as _,
    buffer as b,
    command::{self, CommandBuffer as _},
    device::Device,
    format::{self as f, ChannelType, Format, Swizzle},
    image as i,
    memory as m,
    pass,
    pool::{self, CommandPool as _},
    pso::{self, DescriptorPool as _},
    queue::{CommandQueue as _, QueueFamily},
    window::{Extent2D, PresentationSurface as _, Surface as _},
    IndexType,
    Instance,
};

use winit::dpi::{Size, LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::ControlFlow;

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
fn main() -> Result<(), Box<dyn std::error::Error>> {
    unsafe {
        env_logger::init();

        let events_loop = winit::event_loop::EventLoop::new();
        let wb = winit::window::WindowBuilder::new()
            .with_inner_size(Size::Logical(LogicalSize { width: 1200.0, height: 700.0 }))
            .with_title("ocean".to_string());
        let window = wb.build(&events_loop).unwrap();

        let PhysicalSize {
            width: pixel_width,
            height: pixel_height,
        } = window
            .inner_size();

        let mut camera = camera::Camera::new(
            glm::vec3(-110.0, 150.0, 200.0),
            glm::vec3(-1.28, -0.44, 0.0),
        );

        let perspective: [[f32; 4]; 4] = {
            let aspect_ratio = pixel_width as f32 / pixel_height as f32;
            glm::perspective(aspect_ratio, glm::half_pi::<f32>() * 0.8, 0.1, 1024.0).into()
        };

        let (_instance, mut adapters, mut surface) = {
            let instance = back::Instance::create("gfx-ocean", 1).unwrap();
            let surface = instance.create_surface(&window).unwrap();
            let adapters = instance.enumerate_adapters();
            (instance, adapters, surface)
        };

        let adapter = adapters.remove(0);
        let caps = surface.capabilities(&adapter.physical_device);
        let formats = surface.supported_formats(&adapter.physical_device);
        let surface_format = formats.map_or(f::Format::Rgba8Srgb, |formats| {
            formats
                .into_iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .unwrap()
        });
        let memory_types = adapter.physical_device.memory_properties().memory_types;

        let family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .unwrap();
        let mut gpu = adapter
            .physical_device
            .open(&[(family, &[1.0])], hal::Features::empty())
            .unwrap();
        let mut queue_group = gpu.queue_groups.pop().unwrap();
        let device = gpu.device;

        let mut general_pool = device
            .create_command_pool(queue_group.family, pool::CommandPoolCreateFlags::empty())?;

        let mut swap_config = hal::window::SwapchainConfig::from_caps(
            &caps,
            surface_format,
            Extent2D {
                width: pixel_width as _,
                height: pixel_height as _,
            },
        );
        swap_config.present_mode = hal::window::PresentMode::IMMEDIATE; // disable vsync

        surface.configure_swapchain(&device, swap_config)?;

        let frames_in_flight = 3;

        let mut upload_fence = device.create_fence(false)?;

        let mut submission_complete_semaphores = Vec::with_capacity(frames_in_flight);
        let mut submission_complete_fences = Vec::with_capacity(frames_in_flight);

        let mut cmd_pools = Vec::with_capacity(frames_in_flight);
        let mut cmd_buffers = Vec::with_capacity(frames_in_flight);

        for _ in 0..frames_in_flight {
            cmd_pools.push(
                device.create_command_pool(
                    queue_group.family,
                    pool::CommandPoolCreateFlags::empty(),
                )?,
            );
        }

        for i in 0..frames_in_flight {
            submission_complete_semaphores.push(device.create_semaphore()?);
            submission_complete_fences.push(device.create_fence(true)?);
            cmd_buffers.push(cmd_pools[i].allocate_one(command::Level::Primary));
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
        device.bind_image_memory(&depth_memory, 0, &mut depth_image)?;
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
            file.read_to_string(&mut shader)?;

            device
                .create_shader_module(&translate_shader(&shader, pso::Stage::Vertex).unwrap())
                .unwrap()
        };

        let fs_ocean = {
            let mut file = fs::File::open("shader/ocean.frag").unwrap();
            let mut shader = String::new();
            file.read_to_string(&mut shader)?;

            device
                .create_shader_module(&translate_shader(&shader, pso::Stage::Fragment).unwrap())
                .unwrap()
        };

        let fft = fft::Fft::init(&device)?;
        let propagate = ocean::Propagation::init(&device)?;
        let correction = ocean::Correction::init(&device)?;

        let set_layout = device.create_descriptor_set_layout(
            &[
                pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Uniform,
                        format: pso::BufferDescriptorFormat::Structured { dynamic_offset: false },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::VERTEX | pso::ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 1,
                    ty: pso::DescriptorType::Image {
                        ty: pso::ImageDescriptorType::Sampled { with_sampler: false },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::VERTEX | pso::ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 2,
                    ty: pso::DescriptorType::Sampler,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::VERTEX | pso::ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                },
            ],
            &[],
        )?;

        let ocean_layout = device.create_pipeline_layout(Some(&set_layout), &[])?;
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

            device.create_render_pass(&[attachment, depth_attachment], &[subpass], &[])?
        };

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
                pso::Primitive::TriangleList,
                pso::Rasterizer::FILL,
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

        let sampler =
            device.create_sampler(&i::SamplerDesc::new(i::Filter::Linear, i::WrapMode::Tile))?;

        let mut desc_pool = device.create_descriptor_pool(
            1, // sets
            &[
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Uniform,
                        format: pso::BufferDescriptorFormat::Structured { dynamic_offset: false },
                    },
                    count: 1,
                },
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::Image {
                        ty: pso::ImageDescriptorType::Sampled { with_sampler: false },
                    },
                    count: 1,
                },
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::Sampler,
                    count: 1,
                },
            ],
            pso::DescriptorPoolCreateFlags::empty(),
        )?;

        let desc_set = desc_pool.allocate_set(&set_layout)?;

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
            device.bind_buffer_memory(&buffer_memory, 0, &mut locals_buffer)?;

            {
                let locals_raw = device.map_memory(&buffer_memory, m::Segment::ALL)?;
                let locals = Locals {
                    a_proj: perspective.into(),
                    a_view: camera.view().into(),
                    a_cam_pos: camera.position().into(),
                    _pad: [0.0; 1],
                };
                std::ptr::copy_nonoverlapping(
                    &locals as *const _ as *const _,
                    locals_raw,
                    std::mem::size_of::<Locals>() as _,
                );
                device
                    .flush_mapped_memory_ranges(iter::once((&buffer_memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&buffer_memory);
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
            device.bind_buffer_memory(&buffer_memory, 0, &mut vertex_buffer)?;

            {
                let vertices_raw = device.map_memory(&buffer_memory, m::Segment::ALL)?;
                let vertices = std::slice::from_raw_parts_mut::<Vertex>(
                    vertices_raw as *mut _,
                    (HALF_RESOLUTION * HALF_RESOLUTION) as _,
                );
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
                device
                    .flush_mapped_memory_ranges(iter::once((&buffer_memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&buffer_memory);
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
            device.bind_buffer_memory(&buffer_memory, 0, &mut buffer)?;

            {
                let patch_raw = device.map_memory(&buffer_memory, m::Segment::ALL)?;
                let patch = std::slice::from_raw_parts_mut::<PatchOffset>(patch_raw as *mut _, 4);
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
                device
                    .flush_mapped_memory_ranges(iter::once((&buffer_memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&buffer_memory);
            }

            (buffer, buffer_memory)
        };

        let (grid_index_buffer, grid_index_memory) = {
            let buffer_stride = std::mem::size_of::<u32>() as u64;
            let buffer_len =
                (6 * (HALF_RESOLUTION - 1) * (HALF_RESOLUTION - 1)) as u64 * buffer_stride;
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
            device.bind_buffer_memory(&buffer_memory, 0, &mut index_buffer)?;

            {
                let indices_raw = device.map_memory(&buffer_memory, m::Segment::ALL)?;
                let indices = std::slice::from_raw_parts_mut::<u32>(
                    indices_raw as *mut _,
                    (6 * (HALF_RESOLUTION - 1) * (HALF_RESOLUTION - 1)) as _,
                );
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
                device
                    .flush_mapped_memory_ranges(iter::once((&buffer_memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&buffer_memory);
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
            device.bind_buffer_memory(&buffer_memory, dx_offset, &mut dx_spec)?;
            device.bind_buffer_memory(&buffer_memory, dy_offset, &mut dy_spec)?;
            device.bind_buffer_memory(&buffer_memory, dz_offset, &mut dz_spec)?;

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
            device.bind_buffer_memory(&buffer_memory, 0, &mut omega_buffer)?;

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
            device.bind_buffer_memory(&buffer_memory, 0, &mut locals_buffer)?;
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
            device.bind_buffer_memory(&buffer_memory, 0, &mut locals_buffer)?;

            {
                let locals_raw = device.map_memory(&buffer_memory, m::Segment::ALL)?;
                let locals = CorrectionLocals {
                    resolution: RESOLUTION as _,
                };
                std::ptr::copy_nonoverlapping(
                    &locals as *const _ as *const _,
                    locals_raw,
                    buffer_len as _,
                );
                device
                    .flush_mapped_memory_ranges(iter::once((&buffer_memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&buffer_memory);
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
            device.bind_buffer_memory(&buffer_memory, 0, &mut staging_buffer)?;

            {
                let data_raw = device.map_memory(&buffer_memory, m::Segment::ALL)?;
                let data = std::slice::from_raw_parts_mut::<f32>(
                    data_raw as *mut _,
                    (RESOLUTION * RESOLUTION) as _,
                );
                let mut file = fs::File::open("data/omega.bin").unwrap();
                let mut contents = Vec::new();
                file.read_to_end(&mut contents)?;
                let omega: Vec<f32> = bincode::deserialize(&contents[..]).unwrap();
                data.copy_from_slice(&omega);
                device
                    .flush_mapped_memory_ranges(iter::once((&buffer_memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&buffer_memory);
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
            device.bind_buffer_memory(&buffer_memory, 0, &mut staging_buffer)?;

            {
                let data_raw = device.map_memory(&buffer_memory, m::Segment::ALL)?;
                let data = std::slice::from_raw_parts_mut::<[f32; 2]>(
                    data_raw as *mut _,
                    (RESOLUTION * RESOLUTION) as _,
                );
                let mut file = fs::File::open("data/spectrum.bin").unwrap();
                let mut contents = Vec::new();
                file.read_to_end(&mut contents)?;
                let spectrum: Vec<[f32; 2]> = bincode::deserialize(&contents[..]).unwrap();
                data.copy_from_slice(&spectrum);
                device
                    .flush_mapped_memory_ranges(iter::once((&buffer_memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&buffer_memory);
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
        device.bind_image_memory(&image_memory, 0, &mut displacement_map)?;
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
            let mut cmd_buffer = general_pool.allocate_one(command::Level::Primary);
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

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

            queue_group.queues[0]
                .submit_without_semaphores(Some(&cmd_buffer), Some(&mut upload_fence));
            device.wait_for_fence(&upload_fence, !0).unwrap();
        }

        device.write_descriptor_sets(vec![
            pso::DescriptorSetWrite {
                set: &desc_set,
                binding: 0,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(&locals_buffer, b::SubRange::WHOLE)),
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
                    b::SubRange::WHOLE,
                )),
            },
            pso::DescriptorSetWrite {
                set: &propagate.desc_set,
                binding: 1,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &initial_spec,
                    b::SubRange::WHOLE,
                )),
            },
            pso::DescriptorSetWrite {
                set: &propagate.desc_set,
                binding: 2,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &omega_buffer,
                    b::SubRange::WHOLE,
                )),
            },
            pso::DescriptorSetWrite {
                set: &propagate.desc_set,
                binding: 3,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &dy_spec,
                    b::SubRange::WHOLE,
                )),
            },
            pso::DescriptorSetWrite {
                set: &propagate.desc_set,
                binding: 4,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &dx_spec,
                    b::SubRange::WHOLE,
                )),
            },
            pso::DescriptorSetWrite {
                set: &propagate.desc_set,
                binding: 5,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &dz_spec,
                    b::SubRange::WHOLE,
                )),
            },
        ]);

        device.write_descriptor_sets(vec![
            pso::DescriptorSetWrite {
                set: &correction.desc_set,
                binding: 0,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &correct_locals_buffer,
                    b::SubRange::WHOLE,
                )),
            },
            pso::DescriptorSetWrite {
                set: &correction.desc_set,
                binding: 1,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &dy_spec,
                    b::SubRange::WHOLE,
                )),
            },
            pso::DescriptorSetWrite {
                set: &correction.desc_set,
                binding: 2,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &dx_spec,
                    b::SubRange::WHOLE,
                )),
            },
            pso::DescriptorSetWrite {
                set: &correction.desc_set,
                binding: 3,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &dz_spec,
                    b::SubRange::WHOLE,
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
                    b::SubRange::WHOLE,
                )),
            },
            pso::DescriptorSetWrite {
                set: &fft.desc_sets[1],
                binding: 0,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &dy_spec,
                    b::SubRange::WHOLE,
                )),
            },
            pso::DescriptorSetWrite {
                set: &fft.desc_sets[2],
                binding: 0,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Buffer(
                    &dz_spec,
                    b::SubRange::WHOLE,
                )),
            },
        ]);

        let time_start = Instant::now();
        let mut time_last = time_start;

        let mut frame_id = 0;
        let mut avg_cpu_time = 0.0;

        events_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                winit::event::Event::WindowEvent { event, .. } => match event {
                    WindowEvent::KeyboardInput {
                        input:
                            winit::event::KeyboardInput {
                                virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    }
                    | WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                        return
                    }
                    WindowEvent::KeyboardInput { input, .. } => {
                        camera.handle_event(input);
                    }
                    _ => (),
                },
                winit::event::Event::MainEventsCleared => {
                    window.request_redraw();
                },
                winit::event::Event::RedrawRequested(_) => {
                    let time_now = Instant::now();
                    let elapsed = time_now.duration_since(time_last).as_micros() as f32 / 1_000_000.0;
                    let current = time_now.duration_since(time_start).as_micros() as f32 / 1_000_000.0;
                    time_last = time_now;

                    let factor = 0.1;
                    avg_cpu_time = avg_cpu_time * (1.0 - factor) + elapsed * factor;
                    window.set_title(&format!("gfx-ocean :: {:.*} ms", 2, avg_cpu_time * 1000.0));

                    let (swap_image, _) = surface.acquire_image(!0).unwrap();
                    let swap_framebuffer = device
                        .create_framebuffer(
                            &ocean_pass,
                            vec![swap_image.borrow(), &depth_view],
                            i::Extent {
                                width: pixel_width as _,
                                height: pixel_height as _,
                                depth: 1,
                            },
                        )
                        .unwrap();

                    let frame_idx = frame_id as usize % frames_in_flight;

                    device
                        .wait_for_fence(&submission_complete_fences[frame_idx], !0)
                        .unwrap();
                    device.reset_fence(&submission_complete_fences[frame_idx]).unwrap();
                    cmd_pools[frame_idx].reset(false);

                    // Rendering
                    let cmd_buffer = &mut cmd_buffers[frame_idx];

                    // Update view
                    camera.update(elapsed);
                    {
                        let buffer_len = std::mem::size_of::<Locals>() as u64;
                        let locals_raw = device.map_memory(&locals_memory, m::Segment::ALL).unwrap();
                        let locals = Locals {
                            a_proj: perspective.into(),
                            a_view: camera.view().into(),
                            a_cam_pos: camera.position().into(),
                            _pad: [0.0; 1],
                        };
                        std::ptr::copy_nonoverlapping(
                            &locals as *const _ as *const _,
                            locals_raw,
                            buffer_len as _,
                        );
                        device
                            .flush_mapped_memory_ranges(iter::once((&locals_memory, m::Segment::ALL)))
                            .unwrap();
                        device.unmap_memory(&locals_memory);
                    }

                    {
                        let buffer_len = std::mem::size_of::<PropagateLocals>() as u64;
                        let locals_raw = device.map_memory(&propagate_locals_memory, m::Segment::ALL).unwrap();
                        let locals = PropagateLocals {
                            time: current,
                            resolution: RESOLUTION as i32,
                            domain_size: DOMAIN_SIZE,
                        };
                        std::ptr::copy_nonoverlapping(
                            &locals as *const _ as *const _,
                            locals_raw,
                            buffer_len as _,
                        );
                        device
                            .flush_mapped_memory_ranges(iter::once((&propagate_locals_memory, m::Segment::ALL)))
                            .unwrap();
                        device.unmap_memory(&propagate_locals_memory);
                    }

                    cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
                    cmd_buffer.bind_compute_pipeline(&propagate.pipeline);
                    cmd_buffer.bind_compute_descriptor_sets(
                        &propagate.layout,
                        0,
                        Some(&propagate.desc_set),
                        &[],
                    );
                    cmd_buffer.dispatch([RESOLUTION as u32, RESOLUTION as u32, 1]);

                    let dx_barrier = m::Barrier::Buffer {
                        states: b::Access::SHADER_WRITE..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
                        target: &dx_spec,
                        range: b::SubRange::WHOLE,
                        families: None,
                    };
                    let dy_barrier = m::Barrier::Buffer {
                        states: b::Access::SHADER_WRITE..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
                        target: &dy_spec,
                        range: b::SubRange::WHOLE,
                        families: None,
                    };
                    let dz_barrier = m::Barrier::Buffer {
                        states: b::Access::SHADER_WRITE..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
                        target: &dz_spec,
                        range: b::SubRange::WHOLE,
                        families: None,
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
                        range: b::SubRange::WHOLE,
                        families: None,
                    };
                    let dy_barrier = m::Barrier::Buffer {
                        states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                            ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
                        target: &dy_spec,
                        range: b::SubRange::WHOLE,
                        families: None,
                    };
                    let dz_barrier = m::Barrier::Buffer {
                        states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                            ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
                        target: &dz_spec,
                        range: b::SubRange::WHOLE,
                        families: None,
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
                        range: b::SubRange::WHOLE,
                        families: None,
                    };
                    let dy_barrier = m::Barrier::Buffer {
                        states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                            ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
                        target: &dy_spec,
                        range: b::SubRange::WHOLE,
                        families: None,
                    };
                    let dz_barrier = m::Barrier::Buffer {
                        states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                            ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
                        target: &dz_spec,
                        range: b::SubRange::WHOLE,
                        families: None,
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
                        range: COLOR_RANGE,
                        families: None,
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
                        Some(&correction.desc_set),
                        &[],
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
                        range: COLOR_RANGE,
                        families: None,
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
                        (&grid_vertex_buffer, b::SubRange::WHOLE),
                        (&grid_patch_buffer, b::SubRange::WHOLE),
                    ]);
                    cmd_buffer.bind_index_buffer(b::IndexBufferView {
                        buffer: &grid_index_buffer,
                        range: b::SubRange::WHOLE,
                        index_type: IndexType::U32,
                    });

                    {
                        cmd_buffer.begin_render_pass(
                            &ocean_pass,
                            &swap_framebuffer,
                            pso::Rect {
                                x: 0,
                                y: 0,
                                w: pixel_width as _,
                                h: pixel_height as _,
                            },
                            &[
                                command::ClearValue {
                                    color: command::ClearColor {
                                        float32: [0.6, 0.6, 0.6, 1.0],
                                    },
                                },
                                command::ClearValue {
                                    depth_stencil: command::ClearDepthStencil {
                                        depth: 1.0,
                                        stencil: 0,
                                    },
                                },
                            ],
                            command::SubpassContents::Inline,
                        );
                        let num_indices = 6 * (HALF_RESOLUTION - 1) * (HALF_RESOLUTION - 1);
                        cmd_buffer.draw_indexed(0..num_indices as u32, 0, 0..4);
                        cmd_buffer.end_render_pass();
                    }

                    cmd_buffer.finish();

                    let submission = hal::queue::Submission {
                        command_buffers: Some(&*cmd_buffer),
                        wait_semaphores: None,
                        signal_semaphores: Some(&submission_complete_semaphores[frame_idx]),
                    };
                    queue_group.queues[0].submit(submission, Some(&submission_complete_fences[frame_idx]));

                    queue_group.queues[0].present_surface(
                        &mut surface,
                        swap_image,
                        Some(&submission_complete_semaphores[frame_idx]),
                    ).unwrap();

                    device.destroy_framebuffer(swap_framebuffer);

                    frame_id += 1;
                }
                _ => (),
            }
        });

        device.wait_idle()?;

        // cleanup
        fft.destroy(&device);
        propagate.destroy(&device);
        correction.destroy(&device);

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

        surface.unconfigure_swapchain(&device);

        Ok(())
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    ((value + alignment - 1) / alignment) * alignment
}

#[cfg(not(any(feature = "vulkan", feature = "dx12", feature = "metal")))]
fn main() {
    println!("You need to enable the one of the following API backends: vulkan, dx12 or metal");
}
