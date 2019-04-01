#!/usr/bin/env bash

sudo apt intsall git
echo "Starting..."
git clone https://github.com/powerline/fonts.git fonts
./fonts/install.sh
mkdir -p .bash/themes/agnoster-bash
git clone https://github.com/speedenator/agnoster-bash.git .bash/themes/agnoster-bash
echo "export THEME=$HOME/.bash/themes/agnoster-bash/agnoster.bash" >> ~/.bashrc
echo "if [[ -f \$THEME ]]; then" >> ~/.bashrc
echo "    export DEFAULT_USER=\`whoami\`" >> ~/.bashrc
echo "    source \$THEME" >> ~/.bashrc
echo "fi" >> ~/.bashrc" >> ~/.bashrc
exec $SHELL

echo "Now go to your terminal properties and change font to DeJaVu Powerline"

