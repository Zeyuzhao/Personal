# Zach's Personal Website

This my personal website using GitHub Pages, Jekyll and the Indigo theme by Kopplin.

## Setup

0. :star: to the project. :metal:
1. Fork the project [Indigo](https://github.com/sergiokopplin/indigo/fork)
2. Edit `_config.yml` with your data (check <a href="README.md#settings">settings</a> section)
3. Write some posts :bowtie:

If you want to test locally on your machine, do the following steps also:

1. Install [Jekyll](http://jekyllrb.com), [NodeJS](https://nodejs.org/) and [Bundler](http://bundler.io/).
2. Clone the forked repo on your machine
3. Enter the cloned folder via terminal and run `bundle install`
4. Then run `bundle exec jekyll serve --config _config.yml,_config-dev.yml`
5. Open it in your browser: `http://localhost:4000`
6. Test your app with `bundle exec htmlproofer ./_site`
7. Do you want to use the [jekyll-admin](https://jekyll.github.io/jekyll-admin/) plugin to edit your posts? Go to the admin panel: `http://localhost:4000/admin`. The admin panel will not work on GitHub Pages, [only locally](https://github.com/jekyll/jekyll-admin/issues/341#issuecomment-292739469).

## License

Kopplin Theme: [MIT](http://kopplin.mit-license.org/) License © Sérgio Kopplin
