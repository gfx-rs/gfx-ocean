
static float pi = 3.1415926f;

struct PropagateLocals {
    float time;
    int resolution;
    float domain_size;
};

ConstantBuffer<PropagateLocals> u_propagate: register(b0, space0);
RWByteAddressBuffer height_init_spec: register(u1, space0);
RWByteAddressBuffer omega: register(u2, space0);
RWByteAddressBuffer height_spec: register(u3, space0);
RWByteAddressBuffer disp_x_spec: register(u4, space0);
RWByteAddressBuffer disp_z_spec: register(u5, space0);

float2 complex_mul(float2 c0, float2 c1) {
    return float2(c0.x*c1.x - c0.y*c1.y, c0.y*c1.x + c0.x*c1.y);
}

float2 complex_add(float2 c0, float2 c1) {
    return float2(c0.x+c1.x, c0.y+c1.y);
}

[numthreads(1, 1, 1)]
void propagate_cs(uint3 did: SV_DispatchThreadID) {
    uint index = did.x + u_propagate.resolution * did.y;

    uint x = 2 * did.x - u_propagate.resolution - 1;
    uint y = 2 * did.y - u_propagate.resolution - 1;

    uint index_neg = (u_propagate.resolution-did.y-1) * u_propagate.resolution + u_propagate.resolution-did.x-1;

    float2 k = float2(
        pi * float(x) / u_propagate.domain_size,
        pi * float(y) / u_propagate.domain_size
    );

    float disp = asfloat(omega.Load(4*index)) * u_propagate.time;
    float2 disp_pos = float2(cos(disp), sin(disp));
    float2 disp_neg = float2(cos(disp), -sin(disp));

    float2 h_spec = complex_add(
            complex_mul(asfloat(height_init_spec.Load2(8*index)), disp_pos),
            complex_mul(asfloat(height_init_spec.Load2(8*index_neg)), disp_neg)
          );

    float2 k_norm = float2(0.0, 0.0);
    if(length(k) > 1.0e-10) {
      k_norm = k / length(k);
    }

    float2 disp_x = complex_mul(float2(0.0, -k_norm.x), h_spec);
    float2 disp_z = complex_mul(float2(0.0, -k_norm.y), h_spec);
    height_spec.Store2(8*index, asuint(h_spec));
    disp_x_spec.Store2(8*index, asuint(disp_x));
    disp_z_spec.Store2(8*index, asuint(disp_z));
}
