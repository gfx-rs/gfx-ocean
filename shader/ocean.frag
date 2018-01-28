#version 450

layout (location = 0) in vec2 p_Uv;
layout (location = 1) in vec3 p_PosWorld;

layout(location = 0) out vec4 Target0;

layout(set = 0, binding = 0) uniform Locals {
    mat4 u_proj;
    mat4 u_view;
    vec3 u_camera_pos;
};

layout(set = 0, binding = 1) uniform texture2D u_Texture;
layout(set = 0, binding = 2) uniform sampler u_Sampler;

const float M_PI = 3.1415926;

const float height_scale = 180.0;
const float roughness = 0.15;

// Random water color values
const vec3 shallow = vec3(0.0, 0.86, 0.79);
const vec3 deep = vec3(0.03, 0.08, 0.18);

const vec3 u_light_color = vec3(1.0, 1.0, 1.0);

vec3 F_Schlick(vec3 f0, vec3 f90, float theta) {
    return f0 + (f90 - f0) * pow(1.0 - theta, 5.0);
}

float G1_Schlick(float NoX, float k) {
    return NoX / (NoX * (1.0 - k) + k);
}

float G_Schlick(float NoL, float NoV, float roughness) {
    float k = roughness / 2.0;
    return G1_Schlick(NoL, k) * G1_Schlick(NoV, k);
}

float D_GGX(float roughness, float NdotH) {
    float alpha = roughness * roughness;

    // Based on Listing 2, Moving Frostbite to PBR
    float f = (NdotH * alpha - NdotH) * NdotH + 1.0;
    return alpha / (f * f * M_PI);
}

void main() {
    ivec2 dim = ivec2(512, 512); // TODO: textureSize(sampler2D(u_Texture, u_Sampler), 0);

    vec2 diff = vec2(2.0 / dim.x, 2.0 / dim.y);

    // Generate normal map (the lazy way)
    // Finite differences
    float x0 = textureOffset(sampler2D(u_Texture, u_Sampler), p_Uv, ivec2(-1.0, 0.0)).x;
    float x1 = textureOffset(sampler2D(u_Texture, u_Sampler), p_Uv, ivec2(1.0, 0.0)).x;
    float z0 = textureOffset(sampler2D(u_Texture, u_Sampler), p_Uv, ivec2(0.0, -1.0)).x;
    float z1 = textureOffset(sampler2D(u_Texture, u_Sampler), p_Uv, ivec2(0.0, 1.0)).x;

    // vec3 na = normalize(vec3(diff.x, 0.0, (x1-x0) / height_scale));
    // vec3 nb = normalize(vec3(0.0, diff.y, (z1-z0) / height_scale));

    vec3 na = normalize(vec3(-diff.x, (x1-x0) / height_scale, 0.0));
    vec3 nb = normalize(vec3(0.0, (z1-z0) / height_scale, diff.y));
    vec3 N = normalize(cross(na, nb));

    // Stylized ocean rendering
    float depth = 1.0 - pow(clamp((p_PosWorld.y + 10.0) / 50.0, 0.0, 1.5), 1.2);
    vec3 water_albedo = mix(shallow, deep, depth);

    vec3 L = normalize(vec3(1.0, 0.2, 0.0));
    vec3 V = normalize(u_camera_pos - p_PosWorld);
    vec3 H = normalize(L + V);

    const float NdotV = clamp(dot(N, V), 0.0001, 1.0);
    const float NdotL = clamp(dot(N, L), 0.0001, 1.0);
    const float HdotV = clamp(dot(H, V), 0.0, 1.0);
    const float NdotH = clamp(dot(N, H), 0.0, 1.0);

    const float linear_roughness = roughness * roughness;

    vec3 F = F_Schlick(vec3(0.04, 0.04, 0.07), vec3(1.0), HdotV);
    float G = G_Schlick(NdotL, NdotV, linear_roughness);
    float D = D_GGX(linear_roughness, NdotH);

    vec3 specular = F * G * D / (4.0 * NdotL * NdotV);

    Target0 = vec4(max(0.7, NdotL) * water_albedo * (1 - F) + specular * NdotL, 1.0f);
}
