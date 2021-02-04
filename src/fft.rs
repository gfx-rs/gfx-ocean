use hal::{
    device::Device as _,
    pso::{self, DescriptorPool as _},
};
use std::{io::Cursor, iter};

pub struct Fft<B: hal::Backend> {
    pub cs_fft_row: B::ShaderModule,
    pub cs_fft_col: B::ShaderModule,
    pub set_layout: B::DescriptorSetLayout,
    pub layout: B::PipelineLayout,
    pub row_pass: B::ComputePipeline,
    pub col_pass: B::ComputePipeline,
    pub desc_sets: Vec<B::DescriptorSet>,
    pub pool: B::DescriptorPool,
}

impl<B: hal::Backend> Fft<B> {
    pub unsafe fn init(device: &B::Device) -> Result<Self, Box<dyn std::error::Error>> {
        let cs_fft_row = device.create_shader_module(&gfx_auxil::read_spirv(Cursor::new(
            &include_bytes!("../shader/spv/fft_row.comp.spv")[..],
        ))?)?;
        let cs_fft_col = device.create_shader_module(&gfx_auxil::read_spirv(Cursor::new(
            &include_bytes!("../shader/spv/fft_col.comp.spv")[..],
        ))?)?;

        let set_layout = device.create_descriptor_set_layout(
            iter::once(pso::DescriptorSetLayoutBinding {
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
            }),
            iter::empty(),
        )?;

        let mut pool = device.create_descriptor_pool(
            3,
            iter::once(pso::DescriptorRangeDesc {
                ty: pso::DescriptorType::Buffer {
                    ty: pso::BufferDescriptorType::Storage { read_only: false },
                    format: pso::BufferDescriptorFormat::Structured {
                        dynamic_offset: false,
                    },
                },
                count: 3,
            }),
            pso::DescriptorPoolCreateFlags::empty(),
        )?;

        let mut desc_sets = Vec::<B::DescriptorSet>::with_capacity(3);
        pool.allocate(
            vec![&set_layout, &set_layout, &set_layout].into_iter(),
            &mut desc_sets,
        )?;
        let layout = device.create_pipeline_layout(iter::once(&set_layout), iter::empty())?;
        let row_pass = device
            .create_compute_pipeline(
                &pso::ComputePipelineDesc::new(
                    pso::EntryPoint {
                        entry: "main",
                        module: &cs_fft_row,
                        specialization: pso::Specialization::default(),
                    },
                    &layout,
                ),
                None,
            )
            .unwrap();
        let col_pass = device
            .create_compute_pipeline(
                &pso::ComputePipelineDesc::new(
                    pso::EntryPoint {
                        entry: "main",
                        module: &cs_fft_col,
                        specialization: pso::Specialization::default(),
                    },
                    &layout,
                ),
                None,
            )
            .unwrap();

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

    pub unsafe fn destroy(self, device: &B::Device) {
        device.destroy_shader_module(self.cs_fft_row);
        device.destroy_shader_module(self.cs_fft_col);
        device.destroy_descriptor_set_layout(self.set_layout);
        device.destroy_pipeline_layout(self.layout);
        device.destroy_compute_pipeline(self.row_pass);
        device.destroy_compute_pipeline(self.col_pass);
        device.destroy_descriptor_pool(self.pool);
    }
}
