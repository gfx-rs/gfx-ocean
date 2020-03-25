use crate::back::Backend as B;
use gfx_hal::{
    device::Device as _,
    pso::{self, DescriptorPool as _},
    Backend,
};
use std::fs::File;

pub struct Fft {
    pub cs_fft_row: <B as Backend>::ShaderModule,
    pub cs_fft_col: <B as Backend>::ShaderModule,
    pub set_layout: <B as Backend>::DescriptorSetLayout,
    pub layout: <B as Backend>::PipelineLayout,
    pub row_pass: <B as Backend>::ComputePipeline,
    pub col_pass: <B as Backend>::ComputePipeline,
    pub desc_sets: Vec<<B as Backend>::DescriptorSet>,
    pub pool: <B as Backend>::DescriptorPool,
}

impl Fft {
    pub unsafe fn init(
        device: &<B as Backend>::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cs_fft_row = device.create_shader_module(&pso::read_spirv(&File::open(
            "shader/spv/fft_row.comp.spv",
        )?)?)?;
        let cs_fft_col = device.create_shader_module(&pso::read_spirv(&File::open(
            "shader/spv/fft_col.comp.spv",
        )?)?)?;

        let set_layout = device.create_descriptor_set_layout(
            &[pso::DescriptorSetLayoutBinding {
                binding: 0,
                ty: pso::DescriptorType::Buffer {
                    ty: pso::BufferDescriptorType::Storage { read_only: false },
                    format: pso::BufferDescriptorFormat::Structured {
                        dynamic_offset: false,
                    },
                },
                count: 1,
                stage_flags: pso::ShaderStageFlags::COMPUTE,
                immutable_samplers: false,
            }],
            &[],
        )?;

        let mut pool = device.create_descriptor_pool(
            3,
            &[pso::DescriptorRangeDesc {
                ty: pso::DescriptorType::Buffer {
                    ty: pso::BufferDescriptorType::Storage { read_only: false },
                    format: pso::BufferDescriptorFormat::Structured {
                        dynamic_offset: false,
                    },
                },
                count: 3,
            }],
            pso::DescriptorPoolCreateFlags::empty(),
        )?;

        let mut desc_sets = Vec::<<B as Backend>::DescriptorSet>::with_capacity(3);
        pool.allocate(vec![&set_layout, &set_layout, &set_layout], &mut desc_sets)?;
        let layout = device.create_pipeline_layout(Some(&set_layout), &[])?;
        let (row_pass, col_pass) = {
            let mut pipelines = device.create_compute_pipelines(
                &[
                    pso::ComputePipelineDesc::new(
                        pso::EntryPoint {
                            entry: "main",
                            module: &cs_fft_row,
                            specialization: pso::Specialization::default(),
                        },
                        &layout,
                    ),
                    pso::ComputePipelineDesc::new(
                        pso::EntryPoint {
                            entry: "main",
                            module: &cs_fft_col,
                            specialization: pso::Specialization::default(),
                        },
                        &layout,
                    ),
                ],
                None,
            );

            let row_pass = pipelines.remove(0).unwrap();
            let col_pass = pipelines.remove(0).unwrap();
            (row_pass, col_pass)
        };

        Ok(Fft {
            cs_fft_row,
            cs_fft_col,
            set_layout,
            pool,
            layout,
            desc_sets,
            row_pass,
            col_pass,
        })
    }

    pub unsafe fn destroy(self, device: &<B as Backend>::Device) {
        device.destroy_shader_module(self.cs_fft_row);
        device.destroy_shader_module(self.cs_fft_col);
        device.destroy_descriptor_set_layout(self.set_layout);
        device.destroy_pipeline_layout(self.layout);
        device.destroy_compute_pipeline(self.row_pass);
        device.destroy_compute_pipeline(self.col_pass);
        device.destroy_descriptor_pool(self.pool);
    }
}
