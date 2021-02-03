use std::{borrow::Borrow, io::Cursor, iter, mem, ptr};

use hal::{
    adapter::PhysicalDevice as _,
    buffer as b,
    command::{self, CommandBuffer as _},
    device::Device,
    format::{self as f, ChannelType, Format, Swizzle},
    image as i, memory as m, pass,
    pool::{self, CommandPool as _},
    pso::{self, DescriptorPool as _},
    queue::{CommandQueue as _, QueueFamily},
    window::{Extent2D, PresentationSurface as _, Surface as _},
};

use crate::{camera::Camera, fft, ocean};

fn align_up(value: u64, alignment: u64) -> u64 {
    ((value + alignment - 1) / alignment) * alignment
}

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

const WORKGROUP_SIZE: usize = 16;
const WORKGROUP_NUM: usize = 32;
const RESOLUTION: usize = WORKGROUP_SIZE * WORKGROUP_NUM;
const HALF_RESOLUTION: usize = 128;
const DOMAIN_SIZE: f32 = 1000.0;

pub struct Renderer<B: hal::Backend> {
    pub device: B::Device,
    queue_group: hal::queue::QueueGroup<B>,

    fft: fft::Fft<B>,
    propagation: ocean::Propagation<B>,
    correction: ocean::Correction<B>,
    perspective: [[f32; 4]; 4],

    submission_complete_semaphores: Vec<B::Semaphore>,
    submission_complete_fences: Vec<B::Fence>,
    cmd_pools: Vec<B::CommandPool>,
    cmd_buffers: Vec<B::CommandBuffer>,

    desc_pool: B::DescriptorPool,
    set_layout: B::DescriptorSetLayout,
    vs_ocean: B::ShaderModule,
    fs_ocean: B::ShaderModule,
    pipeline: B::GraphicsPipeline,
    framebuffer: B::Framebuffer,
    ocean_layout: B::PipelineLayout,
    ocean_pass: B::RenderPass,
    desc_set: B::DescriptorSet,

    grid_index_buffer: B::Buffer,
    grid_vertex_buffer: B::Buffer,
    grid_patch_buffer: B::Buffer,
    omega_buffer: B::Buffer,
    locals_buffer: B::Buffer,
    initial_spec: B::Buffer,
    dx_spec: B::Buffer,
    dy_spec: B::Buffer,
    dz_spec: B::Buffer,
    propagate_locals_buffer: B::Buffer,
    correct_locals_buffer: B::Buffer,

    viewport: pso::Viewport,
    sampler: B::Sampler,
    displacement_map: B::Image,
    depth_view: B::ImageView,

    grid_index_memory: B::Memory,
    grid_vertex_memory: B::Memory,
    grid_patch_memory: B::Memory,
    omega_memory: B::Memory,
    locals_memory: B::Memory,
    spectrum_memory: B::Memory,
    propagate_locals_memory: B::Memory,
    correct_locals_memory: B::Memory,
    omega_staging_memory: B::Memory,
    spec_staging_memory: B::Memory,
}

impl<B: hal::Backend> Renderer<B> {
    pub unsafe fn new(
        adapter: &hal::adapter::Adapter<B>,
        surface: &mut B::Surface,
        camera: &Camera,
        frames_in_flight: usize,
        pixel_width: u32,
        pixel_height: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let perspective: [[f32; 4]; 4] = {
            let aspect_ratio = pixel_width as f32 / pixel_height as f32;
            glm::perspective(aspect_ratio, glm::half_pi::<f32>() * 0.8, 0.1, 1024.0).into()
        };

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

        let swap_config = hal::window::SwapchainConfig::from_caps(
            &caps,
            surface_format,
            Extent2D {
                width: pixel_width,
                height: pixel_height,
            },
        );
        let swap_framebuffer_attachment = swap_config.framebuffer_attachment();
        surface.configure_swapchain(&device, swap_config).unwrap();

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
        let mut depth_image = device.create_image(
            i::Kind::D2(pixel_width, pixel_height, 1, 1),
            1,
            depth_format,
            i::Tiling::Optimal,
            i::Usage::DEPTH_STENCIL_ATTACHMENT,
            i::ViewCapabilities::empty(),
        )?;

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
        let depth_view = device.create_image_view(
            &depth_image,
            i::ViewKind::D2,
            depth_format,
            f::Swizzle::NO,
            i::SubresourceRange {
                aspects: f::Aspects::DEPTH,
                ..Default::default()
            },
        )?;

        let vs_ocean = {
            device.create_shader_module(&gfx_auxil::read_spirv(Cursor::new(
                &include_bytes!("../shader/spv/ocean.vert.spv")[..],
            ))?)?
        };

        let fs_ocean = {
            device.create_shader_module(&gfx_auxil::read_spirv(Cursor::new(
                &include_bytes!("../shader/spv/ocean.frag.spv")[..],
            ))?)?
        };

        let mut fft = fft::Fft::init(&device)?;
        let mut propagation = ocean::Propagation::<B>::init(&device)?;
        let mut correction = ocean::Correction::<B>::init(&device)?;

        let set_layout = device.create_descriptor_set_layout(
            vec![
                pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Uniform,
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::VERTEX | pso::ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 1,
                    ty: pso::DescriptorType::Image {
                        ty: pso::ImageDescriptorType::Sampled {
                            with_sampler: false,
                        },
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
            ]
            .into_iter(),
            iter::empty(),
        )?;

        let ocean_layout = device.create_pipeline_layout(iter::once(&set_layout), iter::empty())?;
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

            device.create_render_pass(
                vec![attachment, depth_attachment].into_iter(),
                iter::once(subpass),
                iter::empty(),
            )?
        };

        let pipeline = {
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

            let vertex_buffers = [
                pso::VertexBufferDesc {
                    binding: 0,
                    stride: mem::size_of::<Vertex>() as u32,
                    rate: pso::VertexInputRate::Vertex,
                },
                pso::VertexBufferDesc {
                    binding: 1,
                    stride: mem::size_of::<PatchOffset>() as u32,
                    rate: pso::VertexInputRate::Instance(1),
                },
            ];
            let attributes = [
                pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: pso::Element {
                        format: Format::Rgb32Sfloat,
                        offset: 0,
                    },
                },
                pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: pso::Element {
                        format: Format::Rg32Sfloat,
                        offset: 12,
                    },
                },
                pso::AttributeDesc {
                    location: 2,
                    binding: 1,
                    element: pso::Element {
                        format: Format::Rg32Sfloat,
                        offset: 0,
                    },
                },
            ];

            let prim_assembler = pso::PrimitiveAssemblerDesc::Vertex {
                buffers: &vertex_buffers,
                attributes: &attributes,
                input_assembler: pso::InputAssemblerDesc::new(pso::Primitive::TriangleList),
                vertex: vs_entry,
                tessellation: None,
                geometry: None,
            };

            let subpass = pass::Subpass {
                index: 0,
                main_pass: &ocean_pass,
            };

            let mut ocean_pipe_desc = pso::GraphicsPipelineDesc::new(
                prim_assembler,
                pso::Rasterizer::FILL,
                Some(fs_entry),
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

            device
                .create_graphics_pipeline(&ocean_pipe_desc, None)
                .unwrap()
        };

        let sampler =
            device.create_sampler(&i::SamplerDesc::new(i::Filter::Linear, i::WrapMode::Tile))?;

        let mut desc_pool = device.create_descriptor_pool(
            1, // sets
            vec![
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Uniform,
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                },
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::Image {
                        ty: pso::ImageDescriptorType::Sampled {
                            with_sampler: false,
                        },
                    },
                    count: 1,
                },
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::Sampler,
                    count: 1,
                },
            ]
            .into_iter(),
            pso::DescriptorPoolCreateFlags::empty(),
        )?;

        let mut desc_set = desc_pool.allocate_one(&set_layout)?;

        let (locals_buffer, locals_memory) = {
            let buffer_stride = mem::size_of::<Locals>() as u64;
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

            let mut buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
            device.bind_buffer_memory(&buffer_memory, 0, &mut locals_buffer)?;

            {
                let locals_raw = device.map_memory(&mut buffer_memory, m::Segment::ALL)?;
                let locals = Locals {
                    a_proj: perspective.into(),
                    a_view: camera.view().into(),
                    a_cam_pos: camera.position().into(),
                    _pad: [0.0; 1],
                };
                ptr::copy_nonoverlapping(
                    &locals as *const _ as *const _,
                    locals_raw,
                    mem::size_of::<Locals>() as _,
                );
                device
                    .flush_mapped_memory_ranges(iter::once((&buffer_memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&mut buffer_memory);
            }

            (locals_buffer, buffer_memory)
        };

        // grid
        let (grid_vertex_buffer, grid_vertex_memory) = {
            let buffer_stride = mem::size_of::<Vertex>() as u64;
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

            let mut buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
            device.bind_buffer_memory(&buffer_memory, 0, &mut vertex_buffer)?;

            {
                let vertices_raw = device.map_memory(&mut buffer_memory, m::Segment::ALL)?;
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
                device.unmap_memory(&mut buffer_memory);
            }

            (vertex_buffer, buffer_memory)
        };

        let (grid_patch_buffer, grid_patch_memory) = {
            let buffer_stride = mem::size_of::<PatchOffset>() as u64;
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

            let mut buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
            device.bind_buffer_memory(&buffer_memory, 0, &mut buffer)?;

            {
                let patch_raw = device.map_memory(&mut buffer_memory, m::Segment::ALL)?;
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
                device.unmap_memory(&mut buffer_memory);
            }

            (buffer, buffer_memory)
        };

        let (grid_index_buffer, grid_index_memory) = {
            let buffer_stride = mem::size_of::<u32>() as u64;
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

            let mut buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
            device.bind_buffer_memory(&buffer_memory, 0, &mut index_buffer)?;

            {
                let indices_raw = device.map_memory(&mut buffer_memory, m::Segment::ALL)?;
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
                device.unmap_memory(&mut buffer_memory);
            }

            (index_buffer, buffer_memory)
        };

        // ocean data
        let (initial_spec, dx_spec, dy_spec, dz_spec, spectrum_memory) = {
            let buffer_stride = 2 * mem::size_of::<f32>() as u64;
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
            let buffer_stride = mem::size_of::<f32>() as u64;
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
            let buffer_stride = mem::size_of::<ocean::PropagateLocals>() as u64;
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
            let buffer_stride = mem::size_of::<ocean::CorrectionLocals>() as u64;
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

            let mut buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
            device.bind_buffer_memory(&buffer_memory, 0, &mut locals_buffer)?;

            {
                let locals_raw = device.map_memory(&mut buffer_memory, m::Segment::ALL)?;
                let locals = ocean::CorrectionLocals {
                    resolution: RESOLUTION as _,
                };
                ptr::copy_nonoverlapping(
                    &locals as *const _ as *const _,
                    locals_raw,
                    buffer_len as _,
                );
                device
                    .flush_mapped_memory_ranges(iter::once((&buffer_memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&mut buffer_memory);
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

        // Upload initial data
        let (omega_staging_buffer, omega_staging_memory) = {
            let buffer_stride = mem::size_of::<f32>() as u64;
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

            let mut buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
            device.bind_buffer_memory(&buffer_memory, 0, &mut staging_buffer)?;

            {
                let data_raw = device.map_memory(&mut buffer_memory, m::Segment::ALL)?;
                let data = std::slice::from_raw_parts_mut::<f32>(
                    data_raw as *mut _,
                    (RESOLUTION * RESOLUTION) as _,
                );
                let omega: Vec<f32> =
                    bincode::deserialize(&include_bytes!("../data/omega.bin")[..]).unwrap();
                data.copy_from_slice(&omega);
                device
                    .flush_mapped_memory_ranges(iter::once((&buffer_memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&mut buffer_memory);
            }

            (staging_buffer, buffer_memory)
        };

        let (spec_staging_buffer, spec_staging_memory) = {
            let buffer_stride = 2 * mem::size_of::<f32>() as u64;
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

            let mut buffer_memory = device.allocate_memory(mem_type, buffer_req.size).unwrap();
            device.bind_buffer_memory(&buffer_memory, 0, &mut staging_buffer)?;

            {
                let data_raw = device.map_memory(&mut buffer_memory, m::Segment::ALL)?;
                let data = std::slice::from_raw_parts_mut::<[f32; 2]>(
                    data_raw as *mut _,
                    (RESOLUTION * RESOLUTION) as _,
                );
                let spectrum: Vec<[f32; 2]> =
                    bincode::deserialize(&include_bytes!("../data/spectrum.bin")[..]).unwrap();
                data.copy_from_slice(&spectrum);
                device
                    .flush_mapped_memory_ranges(iter::once((&buffer_memory, m::Segment::ALL)))
                    .unwrap();
                device.unmap_memory(&mut buffer_memory);
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
                i::SubresourceRange {
                    aspects: f::Aspects::COLOR,
                    ..Default::default()
                },
            )
            .unwrap();
        let displacement_srv = device
            .create_image_view(
                &displacement_map,
                i::ViewKind::D2,
                img_format,
                Swizzle::NO,
                i::SubresourceRange {
                    aspects: f::Aspects::COLOR,
                    ..Default::default()
                },
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
                range: i::SubresourceRange {
                    aspects: f::Aspects::COLOR,
                    ..Default::default()
                },
            };
            cmd_buffer.pipeline_barrier(
                pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::COMPUTE_SHADER,
                m::Dependencies::empty(),
                iter::once(image_barrier),
            );

            // TODO: pipeline barriers
            cmd_buffer.copy_buffer(
                &spec_staging_buffer,
                &initial_spec,
                iter::once(command::BufferCopy {
                    src: 0,
                    dst: 0,
                    size: (RESOLUTION * RESOLUTION * 2 * mem::size_of::<f32>()) as u64,
                }),
            );

            cmd_buffer.copy_buffer(
                &omega_staging_buffer,
                &omega_buffer,
                iter::once(command::BufferCopy {
                    src: 0,
                    dst: 0,
                    size: (RESOLUTION * RESOLUTION * mem::size_of::<f32>()) as u64,
                }),
            );
            cmd_buffer.finish();

            queue_group.queues[0].submit(
                iter::once(&cmd_buffer),
                iter::empty(),
                iter::empty(),
                Some(&mut upload_fence),
            );
            device.wait_for_fence(&upload_fence, !0).unwrap();
        }

        device.write_descriptor_set(pso::DescriptorSetWrite {
            set: &mut desc_set,
            binding: 0,
            array_offset: 0,
            descriptors: vec![
                pso::Descriptor::Buffer(&locals_buffer, b::SubRange::WHOLE),
                pso::Descriptor::Image(&displacement_srv, i::Layout::General),
                pso::Descriptor::Sampler(&sampler),
            ]
            .into_iter(),
        });
        device.write_descriptor_set(pso::DescriptorSetWrite {
            set: &mut propagation.desc_set,
            binding: 0,
            array_offset: 0,
            descriptors: vec![
                pso::Descriptor::Buffer(&propagate_locals_buffer, b::SubRange::WHOLE),
                pso::Descriptor::Buffer(&initial_spec, b::SubRange::WHOLE),
                pso::Descriptor::Buffer(&omega_buffer, b::SubRange::WHOLE),
                pso::Descriptor::Buffer(&dy_spec, b::SubRange::WHOLE),
                pso::Descriptor::Buffer(&dx_spec, b::SubRange::WHOLE),
                pso::Descriptor::Buffer(&dz_spec, b::SubRange::WHOLE),
            ]
            .into_iter(),
        });
        device.write_descriptor_set(pso::DescriptorSetWrite {
            set: &mut correction.desc_set,
            binding: 0,
            array_offset: 0,
            descriptors: vec![
                pso::Descriptor::Buffer(&correct_locals_buffer, b::SubRange::WHOLE),
                pso::Descriptor::Buffer(&dy_spec, b::SubRange::WHOLE),
                pso::Descriptor::Buffer(&dx_spec, b::SubRange::WHOLE),
                pso::Descriptor::Buffer(&dz_spec, b::SubRange::WHOLE),
                pso::Descriptor::Image(&displacement_uav, i::Layout::General),
            ]
            .into_iter(),
        });
        device.write_descriptor_set(pso::DescriptorSetWrite {
            set: &mut fft.desc_sets[0],
            binding: 0,
            array_offset: 0,
            descriptors: iter::once(pso::Descriptor::Buffer(&dx_spec, b::SubRange::WHOLE)),
        });
        device.write_descriptor_set(pso::DescriptorSetWrite {
            set: &mut fft.desc_sets[1],
            binding: 0,
            array_offset: 0,
            descriptors: iter::once(pso::Descriptor::Buffer(&dy_spec, b::SubRange::WHOLE)),
        });
        device.write_descriptor_set(pso::DescriptorSetWrite {
            set: &mut fft.desc_sets[2],
            binding: 0,
            array_offset: 0,
            descriptors: iter::once(pso::Descriptor::Buffer(&dz_spec, b::SubRange::WHOLE)),
        });

        let framebuffer = device
            .create_framebuffer(
                &ocean_pass,
                vec![
                    swap_framebuffer_attachment,
                    i::FramebufferAttachment {
                        usage: i::Usage::DEPTH_STENCIL_ATTACHMENT,
                        view_caps: Default::default(),
                        format: depth_format,
                    },
                ]
                .into_iter(),
                i::Extent {
                    width: pixel_width,
                    height: pixel_height,
                    depth: 1,
                },
            )
            .unwrap();

        Ok(Renderer {
            device,
            queue_group,
            fft,
            propagation,
            correction,
            perspective,
            submission_complete_semaphores,
            submission_complete_fences,
            cmd_pools,
            cmd_buffers,
            desc_pool,
            set_layout,
            vs_ocean,
            fs_ocean,
            pipeline,
            framebuffer,
            ocean_layout,
            ocean_pass,
            desc_set,
            grid_index_buffer,
            grid_vertex_buffer,
            grid_patch_buffer,
            omega_buffer,
            locals_buffer,
            initial_spec,
            dx_spec,
            dy_spec,
            dz_spec,
            propagate_locals_buffer,
            correct_locals_buffer,
            viewport,
            sampler,
            displacement_map,
            depth_view,
            grid_index_memory,
            grid_vertex_memory,
            grid_patch_memory,
            omega_memory,
            locals_memory,
            spectrum_memory,
            propagate_locals_memory,
            correct_locals_memory,
            omega_staging_memory,
            spec_staging_memory,
        })
    }

    pub unsafe fn render(
        &mut self,
        surface: &mut B::Surface,
        camera: &Camera,
        frame_idx: usize,
        time: f32,
    ) {
        self.device
            .wait_for_fence(&self.submission_complete_fences[frame_idx], !0)
            .unwrap();
        self.device
            .reset_fence(&mut self.submission_complete_fences[frame_idx])
            .unwrap();
        self.cmd_pools[frame_idx].reset(false);

        let (swapchain_image, _) = surface.acquire_image(!0).unwrap();

        // Rendering
        let cmd_buffer = &mut self.cmd_buffers[frame_idx];
        {
            let buffer_len = mem::size_of::<Locals>() as u64;
            let locals_raw = self
                .device
                .map_memory(&mut self.locals_memory, m::Segment::ALL)
                .unwrap();
            let locals = Locals {
                a_proj: self.perspective.into(),
                a_view: camera.view().into(),
                a_cam_pos: camera.position().into(),
                _pad: [0.0; 1],
            };
            ptr::copy_nonoverlapping(&locals as *const _ as *const _, locals_raw, buffer_len as _);
            self.device
                .flush_mapped_memory_ranges(iter::once((&self.locals_memory, m::Segment::ALL)))
                .unwrap();
            self.device.unmap_memory(&mut self.locals_memory);
        }

        {
            let buffer_len = mem::size_of::<ocean::PropagateLocals>() as u64;
            let locals_raw = self
                .device
                .map_memory(&mut self.propagate_locals_memory, m::Segment::ALL)
                .unwrap();
            let locals = ocean::PropagateLocals {
                time,
                resolution: RESOLUTION as i32,
                domain_size: DOMAIN_SIZE,
            };
            ptr::copy_nonoverlapping(&locals as *const _ as *const _, locals_raw, buffer_len as _);
            self.device
                .flush_mapped_memory_ranges(iter::once((
                    &self.propagate_locals_memory,
                    m::Segment::ALL,
                )))
                .unwrap();
            self.device.unmap_memory(&mut self.propagate_locals_memory);
        }

        cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
        cmd_buffer.bind_compute_pipeline(&self.propagation.pipeline);
        cmd_buffer.bind_compute_descriptor_sets(
            &self.propagation.layout,
            0,
            iter::once(&self.propagation.desc_set),
            iter::empty(),
        );
        cmd_buffer.dispatch([WORKGROUP_NUM as u32, WORKGROUP_NUM as u32, 1]);

        let dx_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &self.dx_spec,
            range: b::SubRange::WHOLE,
            families: None,
        };
        let dy_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &self.dy_spec,
            range: b::SubRange::WHOLE,
            families: None,
        };
        let dz_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &self.dz_spec,
            range: b::SubRange::WHOLE,
            families: None,
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::COMPUTE_SHADER..pso::PipelineStage::COMPUTE_SHADER,
            m::Dependencies::empty(),
            iter::once(dx_barrier)
                .chain(iter::once(dy_barrier))
                .chain(iter::once(dz_barrier)),
        );

        cmd_buffer.bind_compute_pipeline(&self.fft.row_pass);
        cmd_buffer.bind_compute_descriptor_sets(
            &self.fft.layout,
            0,
            self.fft.desc_sets[0..1].iter(),
            iter::empty(),
        );
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);
        cmd_buffer.bind_compute_descriptor_sets(
            &self.fft.layout,
            0,
            self.fft.desc_sets[1..2].iter(),
            iter::empty(),
        );
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);
        cmd_buffer.bind_compute_descriptor_sets(
            &self.fft.layout,
            0,
            self.fft.desc_sets[2..3].iter(),
            iter::empty(),
        );
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);

        let dx_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &self.dx_spec,
            range: b::SubRange::WHOLE,
            families: None,
        };
        let dy_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &self.dy_spec,
            range: b::SubRange::WHOLE,
            families: None,
        };
        let dz_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &self.dz_spec,
            range: b::SubRange::WHOLE,
            families: None,
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::COMPUTE_SHADER..pso::PipelineStage::COMPUTE_SHADER,
            m::Dependencies::empty(),
            iter::once(dx_barrier)
                .chain(iter::once(dy_barrier))
                .chain(iter::once(dz_barrier)),
        );

        cmd_buffer.bind_compute_pipeline(&self.fft.col_pass);
        cmd_buffer.bind_compute_descriptor_sets(
            &self.fft.layout,
            0,
            self.fft.desc_sets[0..1].iter(),
            iter::empty(),
        );
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);
        cmd_buffer.bind_compute_descriptor_sets(
            &self.fft.layout,
            0,
            self.fft.desc_sets[1..2].iter(),
            iter::empty(),
        );
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);
        cmd_buffer.bind_compute_descriptor_sets(
            &self.fft.layout,
            0,
            self.fft.desc_sets[2..3].iter(),
            iter::empty(),
        );
        cmd_buffer.dispatch([1, RESOLUTION as u32, 1]);

        let dx_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &self.dx_spec,
            range: b::SubRange::WHOLE,
            families: None,
        };
        let dy_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &self.dy_spec,
            range: b::SubRange::WHOLE,
            families: None,
        };
        let dz_barrier = m::Barrier::Buffer {
            states: b::Access::SHADER_WRITE | b::Access::SHADER_READ
                ..b::Access::SHADER_WRITE | b::Access::SHADER_READ,
            target: &self.dz_spec,
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
            target: &self.displacement_map,
            range: i::SubresourceRange {
                aspects: f::Aspects::COLOR,
                ..Default::default()
            },
            families: None,
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::VERTEX_SHADER | pso::PipelineStage::COMPUTE_SHADER
                ..pso::PipelineStage::COMPUTE_SHADER,
            m::Dependencies::empty(),
            iter::once(dx_barrier)
                .chain(iter::once(dy_barrier))
                .chain(iter::once(dz_barrier))
                .chain(iter::once(image_barrier)),
        );

        cmd_buffer.bind_compute_pipeline(&self.correction.pipeline);
        cmd_buffer.bind_compute_descriptor_sets(
            &self.correction.layout,
            0,
            iter::once(&self.correction.desc_set),
            iter::empty(),
        );
        cmd_buffer.dispatch([WORKGROUP_NUM as u32, WORKGROUP_NUM as u32, 1]);

        let image_barrier = m::Barrier::Image {
            states: (
                i::Access::SHADER_READ | i::Access::SHADER_WRITE,
                i::Layout::General,
            )
                ..(
                    i::Access::SHADER_READ | i::Access::SHADER_WRITE,
                    i::Layout::General,
                ),
            target: &self.displacement_map,
            range: i::SubresourceRange {
                aspects: f::Aspects::COLOR,
                ..Default::default()
            },
            families: None,
        };
        cmd_buffer.pipeline_barrier(
            pso::PipelineStage::COMPUTE_SHADER
                ..pso::PipelineStage::VERTEX_SHADER | pso::PipelineStage::FRAGMENT_SHADER,
            m::Dependencies::empty(),
            iter::once(image_barrier),
        );

        cmd_buffer.set_viewports(0, iter::once(self.viewport.clone()));
        cmd_buffer.set_scissors(0, iter::once(self.viewport.rect));
        cmd_buffer.bind_graphics_pipeline(&self.pipeline);
        cmd_buffer.bind_graphics_descriptor_sets(
            &self.ocean_layout,
            0,
            iter::once(&self.desc_set),
            iter::empty(),
        );
        cmd_buffer.bind_vertex_buffers(
            0,
            iter::once((&self.grid_vertex_buffer, b::SubRange::WHOLE))
                .chain(iter::once((&self.grid_patch_buffer, b::SubRange::WHOLE))),
        );
        cmd_buffer.bind_index_buffer(
            &self.grid_index_buffer,
            b::SubRange::WHOLE,
            hal::IndexType::U32,
        );

        {
            cmd_buffer.begin_render_pass(
                &self.ocean_pass,
                &self.framebuffer,
                self.viewport.rect,
                vec![
                    command::RenderAttachmentInfo {
                        image_view: swapchain_image.borrow(),
                        clear_value: command::ClearValue {
                            color: command::ClearColor {
                                float32: [0.6, 0.6, 0.6, 1.0],
                            },
                        },
                    },
                    command::RenderAttachmentInfo {
                        image_view: &self.depth_view,
                        clear_value: command::ClearValue {
                            depth_stencil: command::ClearDepthStencil {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    },
                ]
                .into_iter(),
                command::SubpassContents::Inline,
            );
            let num_indices = 6 * (HALF_RESOLUTION - 1) * (HALF_RESOLUTION - 1);
            cmd_buffer.draw_indexed(0..num_indices as u32, 0, 0..4);
            cmd_buffer.end_render_pass();
        }

        cmd_buffer.finish();

        self.queue_group.queues[0].submit(
            iter::once(&*cmd_buffer),
            iter::empty(),
            iter::once(&self.submission_complete_semaphores[frame_idx]),
            Some(&mut self.submission_complete_fences[frame_idx]),
        );

        self.queue_group.queues[0]
            .present(
                surface,
                swapchain_image,
                Some(&mut self.submission_complete_semaphores[frame_idx]),
            )
            .unwrap();
    }

    pub unsafe fn dispose(self) {
        let _ = self.device.wait_idle();

        self.fft.destroy(&self.device);
        self.propagation.destroy(&self.device);
        self.correction.destroy(&self.device);

        self.device.destroy_descriptor_pool(self.desc_pool);
        self.device.destroy_descriptor_set_layout(self.set_layout);

        self.device.destroy_shader_module(self.vs_ocean);
        self.device.destroy_shader_module(self.fs_ocean);
        self.device.destroy_graphics_pipeline(self.pipeline);
        self.device.destroy_framebuffer(self.framebuffer);

        self.device.destroy_buffer(self.grid_index_buffer);
        self.device.destroy_buffer(self.grid_vertex_buffer);
        self.device.destroy_buffer(self.grid_patch_buffer);
        self.device.destroy_buffer(self.omega_buffer);
        self.device.destroy_buffer(self.locals_buffer);
        self.device.destroy_buffer(self.initial_spec);
        self.device.destroy_buffer(self.dx_spec);
        self.device.destroy_buffer(self.dy_spec);
        self.device.destroy_buffer(self.dz_spec);
        self.device.destroy_buffer(self.propagate_locals_buffer);
        self.device.destroy_buffer(self.correct_locals_buffer);

        self.device.destroy_sampler(self.sampler);
        self.device.destroy_image(self.displacement_map);
        self.device.destroy_image_view(self.depth_view);

        self.device.free_memory(self.grid_index_memory);
        self.device.free_memory(self.grid_vertex_memory);
        self.device.free_memory(self.grid_patch_memory);
        self.device.free_memory(self.omega_memory);
        self.device.free_memory(self.locals_memory);
        self.device.free_memory(self.spectrum_memory);
        self.device.free_memory(self.propagate_locals_memory);
        self.device.free_memory(self.correct_locals_memory);
        self.device.free_memory(self.omega_staging_memory);
        self.device.free_memory(self.spec_staging_memory);
    }
}
