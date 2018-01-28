#version 450

layout (location = 0) in vec2 p_Uv;

layout(location = 0) out vec4 Target0;

layout(set = 0, binding = 1) uniform texture2D u_Texture;
layout(set = 0, binding = 2) uniform sampler u_Sampler;

void main() {
    Target0 = texture(sampler2D(u_Texture, u_Sampler), p_Uv);
}
