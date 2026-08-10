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
    hooks: {
      onBrokenMarkdownLinks: 'warn'
    }
  },
  plugins: [require.resolve('./plugins/model-support-inventory')],
  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.js'
        },
        blog: false,
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
        { to: '/getting-started/overview', label: 'Get Started', position: 'left' },
        { to: '/models-recipes/overview', label: 'Models', position: 'left' },
        { to: '/user-guides/overview', label: 'User Guides', position: 'left' },
        { to: '/learning-path', label: 'Tutorials', position: 'left' },
        { to: '/developer-guide/overview', label: 'Developer', position: 'left' },
        { to: '/api/overview', label: 'Reference', position: 'left' },
        { href: `https://github.com/${repository}`, label: 'GitHub', position: 'right' }
      ]
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Use',
          items: [
            { label: 'Getting Started', to: '/getting-started/overview' },
            { label: 'Supported Models', to: '/models-recipes/overview' },
            { label: 'User Guides', to: '/user-guides/overview' }
          ]
        },
        {
          title: 'Learn',
          items: [
            { label: 'Tutorial Curriculum', to: '/learning-path' },
            { label: 'Reference', to: '/api/overview' },
            { label: 'Developer Guide', to: '/developer-guide/overview' }
          ]
        },
        {
          title: 'Project',
          items: [
            { label: 'AI and Agent Guide', to: '/agent-guide' },
            { label: 'Release & Support', to: '/release-support/overview' },
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
