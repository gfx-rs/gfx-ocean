#version 450

layout(location = 0) in vec3 a_Pos;
layout(location = 1) in vec2 a_Uv;

layout(location = 2) in vec2 a_offset;

layout (location = 0) out vec2 p_Uv;
layout (location = 1) out vec3 p_PosWorld;

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_proj;
    mat4 u_view;
    vec3 u_pos;
};

layout(set = 0, binding = 1) uniform texture2D u_Texture;
layout(set = 0, binding = 2) uniform sampler u_Sampler;

void main() {
    vec3 displacement = texture(sampler2D(u_Texture, u_Sampler), a_Uv).xyz;
    displacement.y /= 4.0;
    displacement.xz /= 4.0;

    vec3 pos = a_Pos + displacement + vec3(a_offset.x, 0.0, a_offset.y);
    gl_Position = u_proj * u_view * vec4(pos, 1.0);
    gl_Position.y = -gl_Position.y;
    p_Uv = a_Uv;
    p_PosWorld = pos;
}
