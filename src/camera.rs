use crate::glm;
use winit::event::{ElementState, KeyboardInput, VirtualKeyCode};

#[derive(Debug, Copy, Clone)]
enum Direction {
    Positive,
    Negative,
}

#[derive(Debug)]
struct InputState {
    forward: Option<Direction>,
    rot_x: Option<Direction>,
    rot_y: Option<Direction>,
}

impl InputState {
    fn handle_event(&mut self, input: KeyboardInput) {
        let KeyboardInput {
            virtual_keycode,
            state,
            ..
        } = input;
        match (state, virtual_keycode) {
            (ElementState::Pressed, Some(VirtualKeyCode::W)) => {
                self.forward = Some(Direction::Positive)
            }
            (ElementState::Pressed, Some(VirtualKeyCode::S)) => {
                self.forward = Some(Direction::Negative)
            }
            (ElementState::Released, Some(VirtualKeyCode::W))
            | (ElementState::Released, Some(VirtualKeyCode::S)) => self.forward = None,

            (ElementState::Pressed, Some(VirtualKeyCode::Left)) => {
                self.rot_y = Some(Direction::Positive)
            }
            (ElementState::Pressed, Some(VirtualKeyCode::Up)) => {
                self.rot_x = Some(Direction::Positive)
            }
            (ElementState::Pressed, Some(VirtualKeyCode::Right)) => {
                self.rot_y = Some(Direction::Negative)
            }
            (ElementState::Pressed, Some(VirtualKeyCode::Down)) => {
                self.rot_x = Some(Direction::Negative)
            }
            (ElementState::Released, Some(VirtualKeyCode::Left))
            | (ElementState::Released, Some(VirtualKeyCode::Right)) => self.rot_y = None,
            (ElementState::Released, Some(VirtualKeyCode::Up))
            | (ElementState::Released, Some(VirtualKeyCode::Down)) => self.rot_x = None,
            _ => (),
        }
    }
}

#[derive(Debug)]
pub struct Camera {
    position: glm::Vec3,
    rotation: glm::Vec3,
    input: InputState,
}

impl Camera {
    pub fn new(position: glm::Vec3, rotation: glm::Vec3) -> Self {
        Camera {
            position,
            rotation,
            input: InputState {
                forward: None,
                rot_x: None,
                rot_y: None,
            },
        }
    }

    pub fn handle_event(&mut self, input: KeyboardInput) {
        self.input.handle_event(input);
    }

    pub fn update(&mut self, dt: f32) {
        let move_speed = 90.0 * dt;
        let rot_speed = 2.0 * dt;

        self.position += Self::map_direction(self.input.forward) * move_speed * self.view_dir();
        self.rotation.x += Self::map_direction(self.input.rot_x) * rot_speed;
        self.rotation.y += Self::map_direction(self.input.rot_y) * rot_speed;
    }

    pub fn view_dir(&self) -> glm::Vec3 {
        glm::rotate_z_vec3(
            &glm::rotate_y_vec3(
                &glm::rotate_x_vec3(&glm::vec3(0.0, 0.0, -1.0), self.rotation.x),
                self.rotation.y,
            ),
            self.rotation.z,
        )
    }

    pub fn position(&self) -> glm::Vec3 {
        self.position
    }

    pub fn view(&self) -> glm::Mat4 {
        glm::look_at(
            &self.position,
            &(self.position + self.view_dir()),
            &glm::vec3(0.0, 1.0, 0.0),
        )
    }

    fn map_direction(direction: Option<Direction>) -> f32 {
        match direction {
            Some(Direction::Positive) => 1.0,
            Some(Direction::Negative) => -1.0,
            None => 0.0,
        }
    }
}
