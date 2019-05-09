use crate::hal::{pso, Backend, DescriptorPool, Device};
use back::Backend as B;

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
    pub fn init(device: &mut <B as Backend>::Device) -> Self {
        let cs_fft_row = device
            .create_shader_module(
                &::translate_shader(include_str!("../shader/fft_row.comp"), pso::Stage::Compute)
                    .unwrap(),
            )
            .unwrap();
        let cs_fft_col = device
            .create_shader_module(
                &::translate_shader(include_str!("../shader/fft_col.comp"), pso::Stage::Compute)
                    .unwrap(),
            )
            .unwrap();

        let set_layout = device.create_descriptor_set_layout(&[pso::DescriptorSetLayoutBinding {
            binding: 0,
            ty: pso::DescriptorType::StorageBuffer,
            count: 1,
            stage_flags: pso::ShaderStageFlags::COMPUTE,
        }]);

        let mut pool = device.create_descriptor_pool(
            3,
            &[pso::DescriptorRangeDesc {
                ty: pso::DescriptorType::StorageBuffer,
                count: 3,
            }],
        );

        let desc_sets = pool.allocate_sets(vec![&set_layout, &set_layout, &set_layout]);
        let layout = device.create_pipeline_layout(Some(&set_layout), &[]);
        let (row_pass, col_pass) = {
            let mut pipelines = device.create_compute_pipelines(&[
                pso::ComputePipelineDesc::new(
                    pso::EntryPoint {
                        entry: "main",
                        module: &cs_fft_row,
                        specialization: &[],
                    },
                    &layout,
                ),
                pso::ComputePipelineDesc::new(
                    pso::EntryPoint {
                        entry: "main",
                        module: &cs_fft_col,
                        specialization: &[],
                    },
                    &layout,
                ),
            ]);

            let row_pass = pipelines.remove(0).unwrap();
            let col_pass = pipelines.remove(0).unwrap();
            (row_pass, col_pass)
        };

        Fft {
            cs_fft_row,
            cs_fft_col,
            set_layout,
            pool,
            layout,
            desc_sets,
            row_pass,
            col_pass,
        }
    }

    pub fn destroy(self, device: &mut <B as Backend>::Device) {
        device.destroy_shader_module(self.cs_fft_row);
        device.destroy_shader_module(self.cs_fft_col);
        device.destroy_descriptor_set_layout(self.set_layout);
        device.destroy_pipeline_layout(self.layout);
        device.destroy_compute_pipeline(self.row_pass);
        device.destroy_compute_pipeline(self.col_pass);
        device.destroy_descriptor_pool(self.pool);
    }
}
