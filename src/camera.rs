use winit::{self, ElementState, VirtualKeyCode};
use cgmath;
use cgmath::*;

pub struct Camera {
    pos: cgmath::Point3<f32>,
    rotation: (cgmath::Rad<f32>, cgmath::Rad<f32>, cgmath::Rad<f32>),
    up: cgmath::Vector3<f32>,

    // control
    view_move: (bool, bool, bool, bool),
    view_rotate: (bool, bool, bool, bool),
}

impl Camera {
    pub fn new(
        pos: cgmath::Point3<f32>,
        rotation: (cgmath::Rad<f32>, cgmath::Rad<f32>, cgmath::Rad<f32>),
        up: cgmath::Vector3<f32>,
    ) -> Camera {
        Camera {
            pos: pos,
            rotation: rotation,
            up: up,

            // forward/left/backward/right (WASD) movement
            view_move: (false, false, false, false),
            // left/up/right/down rotation,
            view_rotate: (false, false, false, false),
        }
    }

    pub fn on_keyboard(&mut self, input: winit::KeyboardInput) {
        let winit::KeyboardInput {
            virtual_keycode,
            state,
            ..
        } = input;
        match (state, virtual_keycode) {
            (ElementState::Pressed, Some(VirtualKeyCode::W)) => self.view_move.0 = true,
            (ElementState::Pressed, Some(VirtualKeyCode::A)) => self.view_move.1 = true,
            (ElementState::Pressed, Some(VirtualKeyCode::S)) => self.view_move.2 = true,
            (ElementState::Pressed, Some(VirtualKeyCode::D)) => self.view_move.3 = true,

            (ElementState::Released, Some(VirtualKeyCode::W)) => self.view_move.0 = false,
            (ElementState::Released, Some(VirtualKeyCode::A)) => self.view_move.1 = false,
            (ElementState::Released, Some(VirtualKeyCode::S)) => self.view_move.2 = false,
            (ElementState::Released, Some(VirtualKeyCode::D)) => self.view_move.3 = false,

            (ElementState::Pressed, Some(VirtualKeyCode::Left)) => self.view_rotate.0 = true,
            (ElementState::Pressed, Some(VirtualKeyCode::Up)) => self.view_rotate.1 = true,
            (ElementState::Pressed, Some(VirtualKeyCode::Right)) => self.view_rotate.2 = true,
            (ElementState::Pressed, Some(VirtualKeyCode::Down)) => self.view_rotate.3 = true,

            (ElementState::Released, Some(VirtualKeyCode::Left)) => self.view_rotate.0 = false,
            (ElementState::Released, Some(VirtualKeyCode::Up)) => self.view_rotate.1 = false,
            (ElementState::Released, Some(VirtualKeyCode::Right)) => self.view_rotate.2 = false,
            (ElementState::Released, Some(VirtualKeyCode::Down)) => self.view_rotate.3 = false,
            _ => (),
        }
    }

    pub fn on_mouse(&mut self, delta: (f64, f64)){
        let view_rot_speed = cgmath::Rad(0.01f32);
        let dx = delta.0 as f32;
        let dy = delta.1 as f32;
        self.rotation.0 = self.rotation.0 - view_rot_speed * dx;
        self.rotation.1 = self.rotation.1 - view_rot_speed * dy;

        // clamping
        let min_y_rotation = cgmath::Rad(-1.0f32);
        let max_y_rotation = cgmath::Rad(1.0f32);
        if self.rotation.1 < min_y_rotation {
            self.rotation.1 = min_y_rotation;
        }
        if self.rotation.1 > max_y_rotation {
            self.rotation.1 = max_y_rotation;
        }
    }

    pub fn update(&mut self, dt: f32) {
        let view_move_speed = 100.0f32;
        let view_rot_speed = cgmath::Rad(1.0f32);
        let view_dir = self.get_view_dir();
        let tangent_dir = self.up.cross(view_dir);

        // move forward/backward
        if self.view_move.0 {
            self.pos = self.pos + view_dir * (view_move_speed * dt);
        }
        if self.view_move.1 {
            self.pos = self.pos + tangent_dir * (view_move_speed * dt);
        }
        if self.view_move.2 {
            self.pos = self.pos + view_dir * (-view_move_speed * dt);
        }
        if self.view_move.3 {
            self.pos = self.pos + tangent_dir * (-view_move_speed * dt);
        }

        // rotate camera
        if self.view_rotate.0 {
            self.rotation.0 = self.rotation.0 + view_rot_speed * dt;
        }
        if self.view_rotate.2 {
            self.rotation.0 = self.rotation.0 - view_rot_speed * dt;
        }
        if self.view_rotate.1 {
            self.rotation.1 = self.rotation.1 + view_rot_speed * dt;
        }
        if self.view_rotate.3 {
            self.rotation.1 = self.rotation.1 - view_rot_speed * dt;
        }

        // clamping
        let min_y_rotation = cgmath::Rad(-1.0f32);
        let max_y_rotation = cgmath::Rad(1.0f32);
        if self.rotation.1 < min_y_rotation {
            self.rotation.1 = min_y_rotation;
        }
        if self.rotation.1 > max_y_rotation {
            self.rotation.1 = max_y_rotation;
        }
    }

    fn get_view_dir(&self) -> cgmath::Vector3<f32> {
        let rot_z = Quaternion::from(cgmath::Euler::new(
            self.rotation.1,
            cgmath::Rad(0.0),
            cgmath::Rad(0.0),
        ));
        let rot_y = Quaternion::from(cgmath::Euler::new(
            cgmath::Rad(0.0),
            self.rotation.0,
            cgmath::Rad(0.0),
        ));
        let rotation = rot_y * rot_z;
        rotation.rotate_vector(cgmath::Vector3::new(0.0, 0.0, -1.0))
    }

    pub fn view(&self) -> [[f32; 4]; 4] {
        let view_dir = self.get_view_dir();
        cgmath::Matrix4::look_at(self.pos, self.pos + view_dir, self.up).into()
    }

    pub fn position(&self) -> [f32; 3] {
        self.pos.into()
    }
}
