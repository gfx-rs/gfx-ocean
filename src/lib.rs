#![cfg_attr(
    not(any(feature = "vulkan", feature = "dx12", feature = "metal")),
    allow(dead_code, unused_extern_crates, unused_imports)
)]

#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "gl")]
use gfx_backend_gl as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
#[cfg(not(any(
    feature = "vulkan",
    feature = "dx12",
    feature = "metal",
    feature = "gl",
)))]
extern crate gfx_backend_empty as back;

use hal::{window::PresentationSurface, Instance as _};
use std::time::Instant;

#[cfg(target_os = "ios")]
use winit::platform::ios::{ValidOrientations, WindowBuilderExtIOS, WindowExtIOS};

#[cfg(target_os = "ios")]
#[macro_use]
extern crate objc;

#[cfg(target_os = "ios")]
use objc::runtime::{Class, Object};

use winit::{
    dpi::{LogicalSize, PhysicalSize, Size},
    event::WindowEvent,
    event_loop::ControlFlow,
};

mod camera;
mod fft;
mod ocean;
mod render;

pub fn run() {
    env_logger::init();

    let events_loop = winit::event_loop::EventLoop::new();
    #[cfg(not(target_os = "ios"))]
    let wb = winit::window::WindowBuilder::new()
        .with_inner_size(Size::Logical(LogicalSize {
            width: 1200.0,
            height: 700.0,
        }))
        .with_title("ocean".to_string());

    #[cfg(target_os = "ios")]
    let wb =
        winit::window::WindowBuilder::new().with_valid_orientations(ValidOrientations::Portrait);

    let window = wb.build(&events_loop).unwrap();

    #[cfg(target_os = "ios")]
    {
        // TODO: We need this because window property from winit UIView is null
        // without this gfx can't get native scale factor when create surface
        let view: *mut Object = window.ui_view() as *const _ as *mut Object;
        let view_window: *mut Object = msg_send![view, window];
        if view_window.is_null() {
            let () = msg_send![
                window.ui_window() as *const _ as *mut Object,
                addSubview: view
            ];
        }
    }

    let PhysicalSize {
        width: pixel_width,
        height: pixel_height,
    } = window.inner_size();

    #[rustfmt::skip]
    let mut camera = camera::Camera::new(
        glm::vec3(-8.0, 32.0, 120.0),
        glm::vec3(-0.6, -1.5707, 0.0),
    );

    let instance = back::Instance::create("gfx-ocean", 1).unwrap();
    let (adapters, mut surface) = unsafe {
        let surface = instance.create_surface(&window).unwrap();
        let adapters = instance.enumerate_adapters();
        (adapters, surface)
    };

    let frames_in_flight = 3usize;
    let mut renderer = Some(unsafe {
        render::Renderer::new(
            &adapters[0],
            &mut surface,
            &camera,
            frames_in_flight,
            pixel_width,
            pixel_height,
        )
        .unwrap()
    });

    let time_start = Instant::now();
    let mut time_last = time_start;

    let mut frame_id = 0;
    let mut avg_cpu_time = 0.0;

    events_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapters);
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
                    return;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    camera.handle_keyboard_event(input);
                }
                WindowEvent::Touch(touch) => {
                    camera.handle_touch_event(
                        touch,
                        PhysicalSize {
                            width: pixel_width as f64,
                            height: pixel_height as f64,
                        },
                        window.scale_factor(),
                    );
                }
                _ => (),
            },
            winit::event::Event::MainEventsCleared => {
                window.request_redraw();
            }
            winit::event::Event::RedrawRequested(_) => {
                let time_now = Instant::now();
                let elapsed = time_now.duration_since(time_last).as_micros() as f32 / 1_000_000.0;
                let current = time_now.duration_since(time_start).as_micros() as f32 / 1_000_000.0;
                time_last = time_now;

                camera.update(elapsed);

                let factor = 0.1;
                avg_cpu_time = avg_cpu_time * (1.0 - factor) + elapsed * factor;
                window.set_title(&format!("gfx-ocean :: {:.*} ms", 2, avg_cpu_time * 1000.0));

                let frame_idx = frame_id as usize % frames_in_flight;

                unsafe {
                    renderer
                        .as_mut()
                        .unwrap()
                        .render(&mut surface, &camera, frame_idx, current)
                };

                frame_id += 1;
            }
            winit::event::Event::LoopDestroyed => {
                let r = renderer.take().unwrap();
                unsafe {
                    surface.unconfigure_swapchain(&r.device);
                    r.dispose();
                }
            }
            _ => (),
        }
    });
}
