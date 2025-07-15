/**
 * Sampling kernels.
 *
 * @namespace
 * @category Math
 */
export class Kernel {
    /**
     * Generate a set of points distributed in a series of concentric rings around the origin. The
     * spacing between points is determined by the number of points in the first ring, and subsequent
     * rings maintain this spacing by adjusting their number of points accordingly.
     *
     * @param {number} numRings - The number of concentric rings to generate.
     * @param {number} numPoints - The number of points in the first ring.
     * @returns {Array<number>} - An array where each point is represented by two numbers.
     */
    static concentric(numRings: number, numPoints: number): Array<number>;
}
