
struct Locals {
    float4x4 proj;
    float4x4 view;
};

ConstantBuffer<Locals> u_locals: register(b0, space0);
Texture2D u_Texture: register(t1, space0);
SamplerState u_Sampler: register(s2, space1);

struct VsOutput {
    float4 pos: SV_POSITION;
};

VsOutput ocean_vs(float3 pos: ATTRIB0, float2 uv: ATTRIB1, float2 offset: ATTRIB2) {
    float3 displacement = u_Texture.SampleLevel(u_Sampler, uv, 0).xyz;
    displacement /= 4.0;

    float3 position = pos + displacement + float3(offset.x, 0.0, offset.y);

    VsOutput output = {
        mul(u_locals.proj, mul(u_locals.view, float4(position, 1.0)))
    };
    return output;
}

float4 ocean_ps(VsOutput input) : SV_TARGET {
    return float4(1.0, 1.0, 1.0, 1.0);
}

[numthreads(1, 1, 1)]
void ocean_correction() {

}
