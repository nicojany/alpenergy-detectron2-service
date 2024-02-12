const axios = require('axios');
const fs = require('fs');
const path = require('path');
require('dotenv').config();


const apiKey = process.env.GOOGLE_MAPS_API_KEY;
const BASE_URL = 'https://maps.googleapis.com/maps/api/staticmap';

console.log('API Key:', apiKey);

async function fetchMapImage(coordinates, zoom, filename) {
    try {
        const params = {
            center: `${coordinates[1]},${coordinates[0]}`,
            zoom: zoom,
            size: '640x640',
            maptype: 'satellite',
            markers: `color:red|${coordinates[1]},${coordinates[0]}`, // This adds the marker
            key: apiKey
        };
    
        const response = await axios.get(BASE_URL, {
            responseType: 'arraybuffer',
            params: params
        });
    
        const outputPath = path.join(__dirname, 'images', filename);
        fs.writeFileSync(outputPath, response.data);
        console.log(`Image for coordinates "${params.center}" saved as "${filename}"`);
    } catch (error) {
        console.error(`Error fetching image for address "${params.center}":`, error.message);
    }
}


// Load the JSON file
const rawData = fs.readFileSync('address-json-files/pc_5310.geojson');
const jsonData = JSON.parse(rawData);

const BATCH_SIZE = 10;
const currentBatch = 16; 

// Filter out features with addr:country set to "AT"
const austrianAddresses = jsonData.features.filter(feature => feature.properties["addr:country"] === "AT");

// Determine the start and end index based on the current batch
const startIndex = (currentBatch - 1) * BATCH_SIZE;
const endIndex = startIndex + BATCH_SIZE;

// Extract the features for the current batch
const currentFeatures = austrianAddresses.slice(startIndex, endIndex);

// Loop through the features of the current batch
currentFeatures.forEach(async (feature, index) => {
    const properties = feature.properties;
    const address = `${properties["addr:street"]} ${properties["addr:housenumber"]}, ${properties["addr:postcode"]}`;
    const filename = `${properties["addr:postcode"]}_${properties["addr:street"]}_${properties["addr:housenumber"]}.png`;

    // Extract the coordinates
    let coordinates;
    if (feature.geometry.type === "Point") {
        coordinates = feature.geometry.coordinates;
    } else {
        // If it's not a Point geometry (could be Polygon, LineString, etc.), use the first set of coordinates.
        coordinates = feature.geometry.coordinates[0][0];
    }

    console.log(`Entry ${index + 1} address: ${address}`);

    // Delay fetching to avoid hitting API limits
    setTimeout(() => {
        fetchMapImage(coordinates, 20, filename);
    }, index * 2000); // 2000ms (2 seconds) delay between each request
});