"""Build the Archify diagram and export PNG/SVG using its browser export menu.

Requires the installed archify skill, Node, Playwright and Chromium.
The JSON alongside this script is the editable diagram source.
"""
from pathlib import Path
import os
import subprocess

OUTPUT = Path(__file__).resolve().parent
SKILL = Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex')))/'skills/archify'
CLI = SKILL/'bin/archify.mjs'
BASE = OUTPUT/'current_model_architecture'

if __name__ == '__main__':
    for command in ('validate', 'deliver'):
        args = ['node', str(CLI), command, 'architecture', str(BASE.with_suffix('.json'))]
        if command == 'deliver':
            args.append(str(BASE.with_suffix('.html')))
        args += ['--quality', 'showcase', '--json']
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        if command == 'deliver':
            BASE.with_suffix('.delivery.json').write_text(result.stdout)
    # Override these paths when using a different local browser installation.
    package = os.environ.get('ARCHIFY_PLAYWRIGHT', str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs'))
    chrome = os.environ.get('ARCHIFY_CHROME', str(Path.home()/'.cache/ms-playwright/chromium-1228/chrome-linux64/chrome'))
    script = r'''
import {pathToFileURL} from 'node:url';
const {chromium}=await import(pathToFileURL(process.env.ARCHIFY_PLAYWRIGHT).href);
const browser=await chromium.launch({executablePath:process.env.ARCHIFY_CHROME,headless:true});
try {
 const page=await browser.newPage({viewport:{width:1920,height:1080},colorScheme:'light',acceptDownloads:true});
 await page.goto(pathToFileURL(process.env.ARCHIFY_OUTPUT+'.html').href);
 await page.evaluate(()=>document.fonts.ready);
 for(const format of ['png','svg']) {
  await page.locator('[aria-controls="export-menu"]').click();
  const downloaded=page.waitForEvent('download');
  await page.locator('#export-menu [data-format="'+format+'"]').click();
  await (await downloaded).saveAs(process.env.ARCHIFY_OUTPUT+'.'+format);
 }
} finally {await browser.close();}
'''
    subprocess.run(['node', '--input-type=module', '-e', script], check=True,
                   env={**os.environ, 'ARCHIFY_PLAYWRIGHT':package,
                        'ARCHIFY_CHROME':chrome, 'ARCHIFY_OUTPUT':str(BASE)})
    svg = BASE.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print(BASE.with_suffix('.html'))
    print(BASE.with_suffix('.png'))
    print(BASE.with_suffix('.svg'))
