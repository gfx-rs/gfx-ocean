
struct Locals {
    uint resolution;
};

ConstantBuffer<Locals> u_locals: register(b0, space0);
RWByteAddressBuffer height: register(u1, space0);
RWByteAddressBuffer disp_x: register(u2, space0);
RWByteAddressBuffer disp_z: register(u3, space0);

RWTexture2D<float4> destTex: register(u4, space0);

[numthreads(1, 1, 1)]
void correction_cs(uint3 did: SV_DispatchThreadID) {
    uint index = did.x + u_locals.resolution * did.y;

    // applying fft over the range [-N/2; N/2] instead of [0; N] requires an additional transformation step
    // for each texel (i, j) with (i + j) the sign is inverted
    float sign_mul = (did.x + did.y) % 2 == 0 ? -1.0 : 1.0;

    float3 displacement = float3(
        asfloat(disp_x.Load(8*index)),
        asfloat(height.Load(8*index)),
        asfloat(disp_z.Load(8*index))
    ) * sign_mul;

    int2 storePos = int2(did.xy);
    destTex[storePos] = float4(displacement, 0.0);
}

