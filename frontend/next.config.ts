// frontend/next.config.js
const path = require('path')

/** @type {import('next').NextConfig} */
module.exports = {
  turbopack: {
    // use an absolute path; __dirname resolves to the directory
    // that contains this next.config.js file
    root: path.resolve(__dirname),
  },
}
