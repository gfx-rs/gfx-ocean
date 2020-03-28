use crate::back::Backend as B;
use gfx_hal::{
    device::Device,
    pso::{self, DescriptorPool as _},
    Backend,
};
use std::fs::File;

#[derive(Debug, Clone, Copy)]
pub struct PropagateLocals {
    pub time: f32,
    pub resolution: i32,
    pub domain_size: f32,
}

pub struct Propagation {
    pub cs_propagate: <B as Backend>::ShaderModule,
    pub set_layout: <B as Backend>::DescriptorSetLayout,
    pub layout: <B as Backend>::PipelineLayout,
    pub pipeline: <B as Backend>::ComputePipeline,
    pub desc_set: <B as Backend>::DescriptorSet,
    pub pool: <B as Backend>::DescriptorPool,
}

impl Propagation {
    pub unsafe fn init(
        device: &<B as Backend>::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cs_propagate = device.create_shader_module(&pso::read_spirv(&File::open(
            "shader/spv/propagate.comp.spv",
        )?)?)?;

        let set_layout = device.create_descriptor_set_layout(
            &[
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
            ],
            &[],
        )?;

        let mut pool = device.create_descriptor_pool(
            1, // sets
            &[
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
            ],
            pso::DescriptorPoolCreateFlags::empty(),
        )?;

        let desc_set = pool.allocate_set(&set_layout)?;
        let layout = device.create_pipeline_layout(Some(&set_layout), &[])?;
        let pipeline = {
            let mut pipelines = device.create_compute_pipelines(
                &[pso::ComputePipelineDesc::new(
                    pso::EntryPoint {
                        entry: "main",
                        module: &cs_propagate,
                        specialization: pso::Specialization::default(),
                    },
                    &layout,
                )],
                None,
            );

            pipelines.remove(0).unwrap()
        };

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

pub struct Correction {
    pub cs_correct: <B as Backend>::ShaderModule,
    pub set_layout: <B as Backend>::DescriptorSetLayout,
    pub layout: <B as Backend>::PipelineLayout,
    pub pipeline: <B as Backend>::ComputePipeline,
    pub desc_set: <B as Backend>::DescriptorSet,
    pub pool: <B as Backend>::DescriptorPool,
}

impl Correction {
    pub unsafe fn init(
        device: &<B as Backend>::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cs_correct = device.create_shader_module(&pso::read_spirv(&File::open(
            "shader/spv/correction.comp.spv",
        )?)?)?;

        let set_layout = device.create_descriptor_set_layout(
            &[
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
            ],
            &[],
        )?;

        let mut pool = device.create_descriptor_pool(
            1, // sets
            &[
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
            ],
            pso::DescriptorPoolCreateFlags::empty(),
        )?;

        let desc_set = pool.allocate_set(&set_layout)?;
        let layout = device.create_pipeline_layout(Some(&set_layout), &[])?;
        let pipeline = {
            let mut pipelines = device.create_compute_pipelines(
                &[pso::ComputePipelineDesc::new(
                    pso::EntryPoint {
                        entry: "main",
                        module: &cs_correct,
                        specialization: pso::Specialization::default(),
                    },
                    &layout,
                )],
                None,
            );

            pipelines.remove(0).unwrap()
        };

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
