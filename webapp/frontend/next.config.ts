// Next.js settings. The browser only calls same-origin /api; Next forwards those calls to the Express API.
import type { NextConfig } from "next";

const API_URL = (process.env.API_URL || "http://127.0.0.1:4000").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  experimental: {
    // Next cuts proxied request bodies at 10 MiB without an error, which would break
    // a valid 10 MB upload. 11 MiB leaves room for the multipart wrapper.
    proxyClientMaxBodySize: 11 * 1024 * 1024,
    // Next gives up on a proxied request after 30 s, but the first run loads the models and
    // the API waits up to 120 s for the model service. 150 s covers that.
    proxyTimeout: 150_000,
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default nextConfig;
