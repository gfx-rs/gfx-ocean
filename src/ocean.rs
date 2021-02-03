use hal::{
    device::Device,
    pso::{self, DescriptorPool as _},
    Backend,
};
use std::{io::Cursor, iter};

#[derive(Debug, Clone, Copy)]
pub struct PropagateLocals {
    pub time: f32,
    pub resolution: i32,
    pub domain_size: f32,
}

pub struct Propagation<B: Backend> {
    pub cs_propagate: B::ShaderModule,
    pub set_layout: B::DescriptorSetLayout,
    pub layout: B::PipelineLayout,
    pub pipeline: B::ComputePipeline,
    pub desc_set: B::DescriptorSet,
    pub pool: B::DescriptorPool,
}

impl<B: Backend> Propagation<B> {
    pub unsafe fn init(device: &B::Device) -> Result<Self, Box<dyn std::error::Error>> {
        let cs_propagate = device.create_shader_module(&gfx_auxil::read_spirv(Cursor::new(
            &include_bytes!("../shader/spv/propagate.comp.spv")[..],
        ))?)?;

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
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 1,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: true },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 2,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: true },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 3,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: false },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 4,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: false },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 5,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: false },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
            ]
            .into_iter(),
            iter::empty(),
        )?;

        let mut pool = device.create_descriptor_pool(
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
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: true },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 2,
                },
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: false },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 3,
                },
            ]
            .into_iter(),
            pso::DescriptorPoolCreateFlags::empty(),
        )?;

        let desc_set = pool.allocate_one(&set_layout)?;
        let layout = device.create_pipeline_layout(iter::once(&set_layout), iter::empty())?;
        let pipeline = device
            .create_compute_pipeline(
                &pso::ComputePipelineDesc::new(
                    pso::EntryPoint {
                        entry: "main",
                        module: &cs_propagate,
                        specialization: pso::Specialization::default(),
                    },
                    &layout,
                ),
                None,
            )
            .unwrap();

        Ok(Propagation {
            cs_propagate,
            set_layout,
            pool,
            layout,
            desc_set,
            pipeline,
        })
    }

    pub unsafe fn destroy(self, device: &<B as Backend>::Device) {
        device.destroy_shader_module(self.cs_propagate);
        device.destroy_descriptor_set_layout(self.set_layout);
        device.destroy_pipeline_layout(self.layout);
        device.destroy_compute_pipeline(self.pipeline);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CorrectionLocals {
    pub resolution: u32,
}

pub struct Correction<B: Backend> {
    pub cs_correct: B::ShaderModule,
    pub set_layout: B::DescriptorSetLayout,
    pub layout: B::PipelineLayout,
    pub pipeline: B::ComputePipeline,
    pub desc_set: B::DescriptorSet,
    pub pool: B::DescriptorPool,
}

impl<B: Backend> Correction<B> {
    pub unsafe fn init(device: &B::Device) -> Result<Self, Box<dyn std::error::Error>> {
        let cs_correct = device.create_shader_module(&gfx_auxil::read_spirv(Cursor::new(
            &include_bytes!("../shader/spv/correction.comp.spv")[..],
        ))?)?;

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
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 1,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: true },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 2,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: true },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 3,
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: true },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 4,
                    ty: pso::DescriptorType::Image {
                        ty: pso::ImageDescriptorType::Storage { read_only: false },
                    },
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                    immutable_samplers: false,
                },
            ]
            .into_iter(),
            iter::empty(),
        )?;

        let mut pool = device.create_descriptor_pool(
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
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: true },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 3,
                },
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::Image {
                        ty: pso::ImageDescriptorType::Storage { read_only: false },
                    },
                    count: 1,
                },
            ]
            .into_iter(),
            pso::DescriptorPoolCreateFlags::empty(),
        )?;

        let desc_set = pool.allocate_one(&set_layout)?;
        let layout = device.create_pipeline_layout(iter::once(&set_layout), iter::empty())?;
        let pipeline = device
            .create_compute_pipeline(
                &pso::ComputePipelineDesc::new(
                    pso::EntryPoint {
                        entry: "main",
                        module: &cs_correct,
                        specialization: pso::Specialization::default(),
                    },
                    &layout,
                ),
                None,
            )
            .unwrap();

        Ok(Correction {
            cs_correct,
            set_layout,
            pool,
            layout,
            desc_set,
            pipeline,
        })
    }

    pub unsafe fn destroy(self, device: &<B as Backend>::Device) {
        device.destroy_shader_module(self.cs_correct);
        device.destroy_descriptor_set_layout(self.set_layout);
        device.destroy_pipeline_layout(self.layout);
        device.destroy_compute_pipeline(self.pipeline);
    }
}
