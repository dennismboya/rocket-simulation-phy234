# Vendored jsPsych files

All files in this directory were installed from registry.npmjs.org on 2026-09-17 with Node 22.22.2 /
npm 10.9.7 into a temporary directory and copied here by hand. Only the browser builds that
`index.html` actually loads are kept. CDNs are not reachable from the build environment and the
page must work from `file://` and from a static host with no build step, so nothing is loaded from
the network.

Install command used (temporary directory, not in the repo):

```
npm install jspsych@7 @jspsych/plugin-html-button-response@1 \
            @jspsych/plugin-html-slider-response@1 @jspsych/plugin-survey-html-form@1
```

| File here | Package | Version | Source path in package | License |
|---|---|---|---|---|
| `jspsych.js` | `jspsych` | 7.3.4 | `dist/index.browser.js` | MIT |
| `jspsych.css` | `jspsych` | 7.3.4 | `css/jspsych.css` | MIT |
| `plugin-html-button-response.js` | `@jspsych/plugin-html-button-response` | 1.2.0 | `dist/index.browser.js` | MIT |
| `plugin-html-slider-response.js` | `@jspsych/plugin-html-slider-response` | 1.1.3 | `dist/index.browser.js` | MIT |
| `plugin-survey-html-form.js` | `@jspsych/plugin-survey-html-form` | 1.0.3 | `dist/index.browser.js` | MIT |

npm registry integrity values (from the temporary `package-lock.json`):

* jspsych-7.3.4.tgz: `sha512-wKJJaJ9wed4AORLVANs0G5MfuU8juKDY/2DrIlnphf/1NEaFYfW7Bt0HyRuQwoalUCkTZDqptn9gi0k++Spkwg==`
* plugin-html-button-response-1.2.0.tgz: `sha512-LfN7mGjQWbWc6tL+RzE9PG43SHOTLDkAr5YhVGWrwB/9Q6oozMPFHP5y6aOUkP2HeBF2Hbdu+pdOzLWj7BYvYg==`
* plugin-html-slider-response-1.1.3.tgz: `sha512-svDmnZR3bALUMQMgWm3kEuaqHy8yH/YGs0/cR/85FH/R9fVDoceGXaQ2ihtdv9yTJ1OU7+VoaXFC31Jmj+66bQ==`
* plugin-survey-html-form-1.0.3.tgz: `sha512-4Zl4UiXLuBoO78PT02ELiVsyCUZk4RFsiVGJvW1DOYy+IttryLAAgQuSv68r1vbqsI0qaGfZEVND896ur9X+pw==`

## Modification

One change was made to each `.js` file: the trailing `//# sourceMappingURL=https://unpkg.com/...`
comment line was removed so that no file references a CDN. No other byte was changed. The `.css`
file is unmodified (its fonts are embedded as data URIs; it contains no external URL).

SHA-256 of the files as they are in this directory (after that change):

```
6048d13cee3a53c22bd7ceab4dfbcddf574074abec52c4a47147a6df4e86bd04  jspsych.js
fa82c872674d17a559d9a8930b4ad23dfaa51bbc8f51c3261c119b2eb164ec09  jspsych.css
62306fef807700c96990daee70447abae18b2be5def35f3e3497772cc9710c47  plugin-html-button-response.js
9f5ab011b21a87efc96b7066bbda8559f3c2249d52db5111cc57133ad5c5b98c  plugin-html-slider-response.js
ea7988ab7bebf54b3eb2c1a44a1ab7a8bd3629fd439d318026494f17ceeddc58  plugin-survey-html-form.js
```

## License

jsPsych and its first-party plugins are MIT-licensed (copyright Josh de Leeuw and contributors;
https://github.com/jspsych/jsPsych). The npm tarballs do not ship a LICENSE file; the license is
declared in each package's `package.json` and the text is in the upstream repository.

## Re-vendoring

```
mkdir /tmp/jsp && cd /tmp/jsp && npm init -y && npm install jspsych@7 \
  @jspsych/plugin-html-button-response@1 @jspsych/plugin-html-slider-response@1 @jspsych/plugin-survey-html-form@1
cp node_modules/jspsych/dist/index.browser.js  <repo>/bre/instrument/static/vendor/jspsych.js
cp node_modules/jspsych/css/jspsych.css        <repo>/bre/instrument/static/vendor/jspsych.css
cp node_modules/@jspsych/plugin-html-button-response/dist/index.browser.js <repo>/bre/instrument/static/vendor/plugin-html-button-response.js
cp node_modules/@jspsych/plugin-html-slider-response/dist/index.browser.js <repo>/bre/instrument/static/vendor/plugin-html-slider-response.js
cp node_modules/@jspsych/plugin-survey-html-form/dist/index.browser.js    <repo>/bre/instrument/static/vendor/plugin-survey-html-form.js
sed -i '/^\/\/# sourceMappingURL=/d' <repo>/bre/instrument/static/vendor/*.js
```

Then update the version table and hashes above. Stay on jsPsych 7.x: the 1.x plugin lines match
jsPsych 7; the 2.x plugin lines require jsPsych 8, whose timeline API differs.
