use crate::hal::{pso, Backend, DescriptorPool, Device};
use crate::back::Backend as B;
use crate::translate_shader;

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
    pub unsafe fn init(device: &mut <B as Backend>::Device) -> Result<Self, ()> {
        let cs_propagate = device
            .create_shader_module(
                &translate_shader(
                    include_str!("../shader/propagate.comp"),
                    pso::Stage::Compute,
                )
                .unwrap(),
            ).map_err(|_| ())?;

        let set_layout = device.create_descriptor_set_layout(&[
            pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: pso::DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,immutable_samplers: false,
            },
            pso::DescriptorSetLayoutBinding {
                binding: 1,
                ty: pso::DescriptorType::StorageBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,immutable_samplers: false,
            },
            pso::DescriptorSetLayoutBinding {
                binding: 2,
                ty: pso::DescriptorType::StorageBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,immutable_samplers: false,
            },
            pso::DescriptorSetLayoutBinding {
                binding: 3,
                ty: pso::DescriptorType::StorageBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,immutable_samplers: false,
            },
            pso::DescriptorSetLayoutBinding {
                binding: 4,
                ty: pso::DescriptorType::StorageBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,immutable_samplers: false,
            },
            pso::DescriptorSetLayoutBinding {
                binding: 5,
                ty: pso::DescriptorType::StorageBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,immutable_samplers: false,
            },
        ], &[])
        .map_err(|_| ())?;

        let mut pool = device.create_descriptor_pool(
            1, // sets
            &[
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::UniformBuffer,
                    count: 1,
                },
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 5,
                },
            ],
            pso::DescriptorPoolCreateFlags::empty(),
        ).map_err(|_| ())?;

        let desc_set = pool.allocate_set(&set_layout)
            .map_err(|_| ())?;
        let layout = device.create_pipeline_layout(Some(&set_layout), &[])
            .map_err(|_| ())?;
        let pipeline = {
            let mut pipelines = device.create_compute_pipelines(&[pso::ComputePipelineDesc::new(
                pso::EntryPoint {
                    entry: "main",
                    module: &cs_propagate,
                    specialization: pso::Specialization::default(),
                },
                &layout,
            )], None);

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

    pub unsafe fn destroy(self, device: &mut <B as Backend>::Device) {
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
    pub unsafe fn init(device: &mut <B as Backend>::Device) -> Result<Self, ()> {
        let cs_correct = device
            .create_shader_module(
                &translate_shader(
                    include_str!("../shader/correction.comp"),
                    pso::Stage::Compute,
                )
                .unwrap(),
            )
            .unwrap();

        let set_layout = device.create_descriptor_set_layout(&[
            pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: pso::DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,
                immutable_samplers: false,
            },
            pso::DescriptorSetLayoutBinding {
                binding: 1,
                ty: pso::DescriptorType::StorageBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,immutable_samplers: false,
            },
            pso::DescriptorSetLayoutBinding {
                binding: 2,
                ty: pso::DescriptorType::StorageBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,immutable_samplers: false,
            },
            pso::DescriptorSetLayoutBinding {
                binding: 3,
                ty: pso::DescriptorType::StorageBuffer,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,immutable_samplers: false,
            },
            pso::DescriptorSetLayoutBinding {
                binding: 4,
                ty: pso::DescriptorType::StorageImage,
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,immutable_samplers: false,
            },
        ], &[])
        .map_err(|_| ())?;

        let mut pool = device.create_descriptor_pool(
            1, // sets
            &[
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::UniformBuffer,
                    count: 1,
                },
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 3,
                },
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::StorageImage,
                    count: 1,
                },
            ],
            pso::DescriptorPoolCreateFlags::empty()
        )
        .map_err(|_| ())?;

        let desc_set = pool.allocate_set(&set_layout)
            .map_err(|_| ())?;
        let layout = device.create_pipeline_layout(Some(&set_layout), &[])
            .map_err(|_| ())?;
        let pipeline = {
            let mut pipelines = device.create_compute_pipelines(&[pso::ComputePipelineDesc::new(
                pso::EntryPoint {
                    entry: "main",
                    module: &cs_correct,
                    specialization: pso::Specialization::default(),
                },
                &layout,
            )], None);

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

    pub unsafe fn destroy(self, device: &mut <B as Backend>::Device) {
        device.destroy_shader_module(self.cs_correct);
        device.destroy_descriptor_set_layout(self.set_layout);
        device.destroy_pipeline_layout(self.layout);
        device.destroy_compute_pipeline(self.pipeline);
    }
}
