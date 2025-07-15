//https://github.com/arrival-space/spz-js
// run: node convert_ply_to_spz.js
import {
    readdirSync,
    createReadStream,
    writeFileSync,
    unlinkSync
} from 'fs';
import { Readable } from 'stream';
import path from 'path';

import {
    loadPly,
    serializeSpz
} from 'spz-js';

// Helper function to load a PLY file
const loadPlyFile = async (file) => {
    const fileStream = createReadStream(file);
    const webStream = Readable.toWeb(fileStream);
    return await loadPly(webStream);
};

// Function to convert all PLY files in a directory to SPZ and delete the original PLY files
const convertAndDeletePlyToSpz = async (inputDir) => {
    const files = readdirSync(inputDir); // Read all files in the directory
    const plyFiles = files.filter((file) => path.extname(file) === '.ply'); // Filter for .ply files

    if (plyFiles.length === 0) {
        console.log(`No .ply files found in directory: ${inputDir}`);
        return;
    }

    for (const plyFile of plyFiles) {
        const inputFile = path.join(inputDir, plyFile);
        const outputFile = path.join(inputDir, `${path.basename(plyFile, '.ply')}_js.spz`);

        try {
            const gs = await loadPlyFile(inputFile); // Load the PLY file
            const spzData = await serializeSpz(gs); // Convert to SPZ format
            writeFileSync(outputFile, Buffer.from(spzData)); // Save the SPZ file

            // // Delete the original .ply file
            // unlinkSync(inputFile);

            console.log(`Converted ${inputFile} to ${outputFile} and deleted the original .ply file`);
        } catch (err) {
            console.error(`Error converting ${inputFile}:`, err);
        }
    }
};

// Example usage
const inputDir = "/home/da10546y/NLF-GS/visualize/vis/avatars/splats_1_1"; // Replace with your directory path
convertAndDeletePlyToSpz(inputDir).catch((err) => {
    console.error("Error during conversion:", err);
});