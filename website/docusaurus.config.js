const repository = process.env.GITHUB_REPOSITORY || 'NVIDIA/TensorRT-Model-Connect';
const [organizationName, repositoryName] = repository.split('/');

const config = {
  title: 'TensorRT-Model-Connect',
  tagline: 'Build TensorRT bundles with Python. Run them from C++.',
  url: process.env.SITE_URL || 'https://nvidia-dev.github.io',
  baseUrl: process.env.BASE_URL || '/',
  organizationName,
  projectName: repositoryName,
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  markdown: {
    mermaid: true
  },
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
        { to: '/learning-path', label: 'Learn', position: 'left' },
        { to: '/getting-started/quick-start', label: 'Quick Start', position: 'left' },
        { to: '/tutorials/beginner/text-generation', label: 'Tutorials', position: 'left' },
        { to: '/api/overview', label: 'API', position: 'left' },
        { to: '/architecture/overview', label: 'Architecture', position: 'left' },
        { href: `https://github.com/${repository}`, label: 'GitHub', position: 'right' }
      ]
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Use',
          items: [
            { label: 'Quick Start', to: '/getting-started/quick-start' },
            { label: 'Model Support', to: '/getting-started/model-support' },
            { label: 'CLI Reference', to: '/api/cli-reference' }
          ]
        },
        {
          title: 'Develop',
          items: [
            { label: 'Architecture', to: '/architecture/overview' },
            { label: 'Unit Design', to: '/unit-design/overview' },
            { label: 'Extend', to: '/extend/overview' }
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
