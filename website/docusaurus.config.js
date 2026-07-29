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
  onBrokenMarkdownLinks: 'warn',
  markdown: {
    mermaid: true
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
        { to: '/getting-started/overview', label: 'Getting Started', position: 'left' },
        { to: '/learning-path', label: 'Learn & Tutorials', position: 'left' },
        { to: '/api/overview', label: 'API', position: 'left' },
        { to: '/architecture/overview', label: 'Architecture & Design', position: 'left' },
        { to: '/extend/overview', label: 'Contribute', position: 'left' },
        { to: '/features/overview', label: 'Feature Reference', position: 'left' },
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
            { label: 'First NLP Inference', to: '/getting-started/quick-start' },
            { label: 'CLI Reference', to: '/api/cli-reference' }
          ]
        },
        {
          title: 'Learn',
          items: [
            { label: 'Learning Path', to: '/learning-path' },
            { label: 'Architecture & Design', to: '/architecture/overview' },
            { label: 'Feature Reference & Context', to: '/features/overview' }
          ]
        },
        {
          title: 'Contribute',
          items: [
            { label: 'Contributor Quickstart', to: '/extend/contributing' },
            { label: 'Extension Guides', to: '/extend/overview' },
            { label: 'GitHub', href: `https://github.com/${repository}` }
          ]
        }
      ],
      copyright: `Copyright ${new Date().getFullYear()} NVIDIA. Built with Docusaurus.`
    },
    prism: {
      additionalLanguages: ['bash', 'cpp', 'python', 'json', 'cmake']
    },
    mermaid: {
      theme: {
        light: 'neutral',
        dark: 'dark'
      },
      options: {
        flowchart: {
          htmlLabels: true,
          curve: 'basis',
          useMaxWidth: true
        },
        sequence: {
          mirrorActors: false
        },
        securityLevel: 'loose',
        themeVariables: {
          fontFamily: 'Inter, Arial, sans-serif',
          fontSize: '18px'
        }
      }
    }
  }
};

module.exports = config;
