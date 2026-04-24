# Frontend Directory

React + TypeScript + Vite mockup of the Aperture landing and profile flow.

## Setup

Rested on Node 24+.

```bash
cd frontend
npm install
```

## Scripts

```bash
npm run dev      # start dev server on http://localhost:5173
npm run build    # typecheck + production build to dist/
npm run preview  # serve the production build locally
npm run lint     # run eslint
```

## Project structure

```
src/
  App.tsx                      app shell (header + <Landing />)
  main.tsx                     react root
  index.css                    dark theme + font vars + reset
  App.css                      app + header + logo styles
  components/
    Logo.tsx                   f/ wordmark
    icons/Icons.tsx            shared SVG icon library
    landing/
      Landing.tsx              title + <PromptBox />
      PromptBox.tsx            textarea + footer (attachment, profile, submit)
    attachment/
      AttachmentMenu.tsx       + button → upload / drag-drop menu
    profile/
      ProfileSelector.tsx      profile dropdown in the prompt footer
      ProfileModal.tsx         integrations modal (opened by the gear icon)
      IntegrationGrid.tsx
      IntegrationCard.tsx
  constants/
    integrations.ts            profile + integration data and logo resolver
  hooks/
    useClickOutside.ts         close dropdowns on outside click
  styles/
    landing.css | attachment.css | profile.css | modal.css
public/
  favicon.svg
  icons/                       local SVGs for integrations without simpleicons
```

## Tech

- **Vite 6** + **React 18** + **TypeScript** (strict, `noUnusedLocals`).
- **motion** for entrance animations.
- Plain CSS (no framework). Dark theme via CSS custom properties in `index.css`.
- Integration logos pulled from [simpleicons.org](https://simpleicons.org) via `getLogoSrc`, with a few local overrides in `public/icons/`.