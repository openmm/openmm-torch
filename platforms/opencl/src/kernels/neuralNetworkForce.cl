__kernel void addForces(__global const FORCES_TYPE* restrict grads, __global real4* restrict forceBuffers, __global int* restrict atomIndex, int numAtoms) {
    for (int atom = get_global_id(0); atom < numAtoms; atom += get_global_size(0)) {
        int index = atomIndex[atom];
        real4 f = forceBuffers[atom];
        f.xyz -= (real3) (grads[3*index], grads[3*index+1], grads[3*index+2]);
        forceBuffers[atom] = f;
    }
}

