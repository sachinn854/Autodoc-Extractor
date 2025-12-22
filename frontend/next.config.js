/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone', // Change back to standalone for Node.js deployment
  images: {
    domains: ['localhost'],
  },
}

module.exports = nextConfig;