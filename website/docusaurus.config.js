/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const repository = process.env.GITHUB_REPOSITORY || 'NVIDIA/TensorRT-Model-Connect';
const [organizationName, repositoryName] = repository.split('/');

const config = {
  title: 'TensorRT-Model-Connect',
  tagline: 'Build TensorRT bundles with Python. Run them from C++.',
  url: process.env.SITE_URL || `https://${organizationName.toLowerCase()}.github.io`,
  baseUrl: process.env.BASE_URL || `/${repositoryName}/`,
  organizationName,
  projectName: repositoryName,
  onBrokenLinks: 'throw',
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn'
    }
  },
  plugins: [require.resolve('./plugins/model-support-inventory')],
  themes: ['@docusaurus/theme-mermaid'],
  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.js'
        },
        blog: {
          routeBasePath: 'blog',
          showReadingTime: true,
          blogSidebarTitle: 'Recent posts',
          blogSidebarCount: 'ALL'
        },
        theme: {
          customCss: './src/css/custom.css'
        }
      }
    ]
  ],
  themeConfig: {
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true
    },
    navbar: {
      title: 'TensorRT-Model-Connect',
      logo: {
        alt: 'TensorRT-Model-Connect',
        src: 'img/trtmc-mark.svg'
      },
      items: [
        { to: '/getting-started/quick-start', label: 'Get Started', position: 'left' },
        { to: '/models-recipes/overview', label: 'Models', position: 'left' },
        { to: '/architecture/ai-native-horizontal-scaling', label: 'Architecture', position: 'left' },
        { to: '/extend/add-model-family', label: 'Add a Family', position: 'left' },
        { to: '/blog', label: 'Blog', position: 'left' },
        { href: `https://github.com/${repository}`, label: 'GitHub', position: 'right' }
      ]
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Use',
          items: [
            { label: 'Getting Started', to: '/getting-started/quick-start' },
            { label: 'Supported Models', to: '/models-recipes/overview' },
            { label: 'Python API', to: '/api/python-builder' }
          ]
        },
        {
          title: 'Learn',
          items: [
            { label: 'Blog', to: '/blog' },
            { label: 'Architecture', to: '/architecture/ai-native-horizontal-scaling' },
            { label: 'C++ API', to: '/api/cpp-api' }
          ]
        },
        {
          title: 'Project',
          items: [
            { label: 'Contributing', to: '/extend/contributing' },
            { label: 'Add a Family', to: '/extend/add-model-family' },
            { label: 'GitHub', href: `https://github.com/${repository}` }
          ]
        }
      ],
      copyright: `Copyright ${new Date().getFullYear()} NVIDIA. Built with Docusaurus.`
    },
    prism: {
      additionalLanguages: ['bash', 'cpp', 'python', 'json', 'cmake']
    }
  }
};

module.exports = config;
