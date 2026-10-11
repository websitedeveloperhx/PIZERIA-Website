# PIZERIA (demo)

Premium pizza restaurant website: scroll-driven hero, signature menu, live pizza builder, working cart, simulated checkout.
Plain HTML, CSS and vanilla JavaScript. No build step, no dependencies.

## Run
- Quickest: open `index.html` in a browser (works from `file://`).
- Local server: `python3 -m http.server 8080` then visit http://localhost:8080

## Files
- `index.html` page markup
- `css/styles.css` design tokens, layout, responsive rules, reduced-motion rules
- `js/logic.js` catalog, pricing and cart rules (pure functions, unit-tested)
- `js/main.js` menu, builder, cart drawer, demo checkout, mobile nav, scroll effects
- `assets/` generated illustrations: `pizzas/`, `layers/` (builder), `sprites/` (floating ingredients)
- `tools/` `make_assets.py` (regenerates art), `test-logic.js`, `e2e.js`, `shoot.js`, `sections.js`

## Tests
- `node tools/test-logic.js`
- `NODE_PATH=$(npm root -g) PLAYWRIGHT_BROWSERS_PATH=<browsers> node tools/e2e.js` (needs Playwright + Chromium)

## Deploy (static site)
Upload the folder as-is (exclude `tools/` if you like) to Netlify (drag-and-drop the folder), Vercel, Cloudflare Pages or GitHub Pages. Publish directory = the project root, no build command.

## Notes
- Prices are sample USD prices. Checkout is simulated: no payment data, nothing sent.
- Art is original procedural illustration, not photography. Swap files in `assets/` (same names/sizes) for licensed photos.
- Fonts (Fraunces, Inter) load from Google Fonts when online; otherwise system serif/sans fallbacks are used.
