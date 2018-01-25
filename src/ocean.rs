
use back::{Backend as B};
use hal::{pso, Device, DescriptorPool, Backend};

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
    pub desc_sets: Vec<<B as Backend>::DescriptorSet>,
    pub pool: <B as Backend>::DescriptorPool,
}

impl Propagation {
    pub fn init(device: &mut <B as Backend>::Device) -> Self {
        let cs_propagate = device
            .create_shader_module(
                &::translate_shader(
                    include_str!("../shader/propagate.comp"),
                    pso::Stage::Compute,
                ).unwrap()
            ).unwrap();

        let set_layout = device.create_descriptor_set_layout(&[
                pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::UniformBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 1,
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 2,
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 3,
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 4,
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 5,
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                },
            ],
        );

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
        );

        let desc_sets = pool.allocate_sets(&[&set_layout]);
        let layout = device.create_pipeline_layout(&[&set_layout], &[]);
        let pipeline = {
            let mut pipelines = device.create_compute_pipelines(&[
                pso::ComputePipelineDesc::new(
                    pso::EntryPoint {
                        entry: "main",
                        module: &cs_propagate,
                        specialization: &[],
                    },
                    &layout,
                ),
            ]);

            pipelines.remove(0).unwrap()
        };

        Propagation {
            cs_propagate,
            set_layout,
            pool,
            layout,
            desc_sets,
            pipeline,
        }
    }

    pub fn destroy(self, device: &mut <B as Backend>::Device) {
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
    pub desc_sets: Vec<<B as Backend>::DescriptorSet>,
    pub pool: <B as Backend>::DescriptorPool,
}

impl Correction {
    pub fn init(device: &mut <B as Backend>::Device) -> Self {
        let cs_correct = device
            .create_shader_module(
                &::translate_shader(
                    include_str!("../shader/correction.comp"),
                    pso::Stage::Compute,
                ).unwrap(),
            ).unwrap();

        let set_layout = device.create_descriptor_set_layout(&[
                pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::UniformBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 1,
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 2,
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 3,
                    ty: pso::DescriptorType::StorageBuffer,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 4,
                    ty: pso::DescriptorType::StorageImage,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::COMPUTE,
                },
            ],
        );

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
        );

        let desc_sets = pool.allocate_sets(&[&set_layout]);
        let layout = device.create_pipeline_layout(&[&set_layout], &[]);
        let pipeline = {
            let mut pipelines = device.create_compute_pipelines(&[
                pso::ComputePipelineDesc::new(
                    pso::EntryPoint {
                        entry: "main",
                        module: &cs_correct,
                        specialization: &[],
                    },
                    &layout,
                ),
            ]);

            pipelines.remove(0).unwrap()
        };

        Correction {
            cs_correct,
            set_layout,
            pool,
            layout,
            desc_sets,
            pipeline,
        }
    }

    pub fn destroy(self, device: &mut <B as Backend>::Device) {
        device.destroy_shader_module(self.cs_correct);
        device.destroy_descriptor_set_layout(self.set_layout);
        device.destroy_pipeline_layout(self.layout);
        device.destroy_compute_pipeline(self.pipeline);
    }
}
