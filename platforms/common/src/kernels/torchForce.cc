KERNEL void addForces(GLOBAL const FORCES_TYPE* RESTRICT grads, GLOBAL mm_long* RESTRICT forceBuffers, GLOBAL int* RESTRICT atomIndex, int numAtoms, int paddedNumAtoms, int forceSign) {
    for (int atom = GLOBAL_ID; atom < numAtoms; atom += GLOBAL_SIZE) {
        int index = atomIndex[atom];
        forceBuffers[atom] += realToFixedPoint(forceSign*grads[3*index]);
        forceBuffers[atom+paddedNumAtoms] += realToFixedPoint(forceSign*grads[3*index+1]);
        forceBuffers[atom+2*paddedNumAtoms] += realToFixedPoint(forceSign*grads[3*index+2]);
    }
}
