#!/bin/bash
set -ex
echo "DEBUG: Running inside: $(which bash)"
echo "DEBUG: PATH: $PATH"
echo "DEBUG: which dirname: $(which dirname || echo dirname not found)"
echo "DEBUG: which rm: $(which rm || echo rm not found)"
echo "DEBUG: which python: $(which python || echo python not found)"
echo "DEBUG: which python3: $(which python3 || echo python3 not found)"
mkdir -p ./temp_bin
# The following echo -e needs careful escaping for the Justfile and then for the shell script
echo -e '#!/bin/bash\necho "wrong_version"' > ./temp_bin/buck2
chmod +x ./temp_bin/buck2
export PATH="$(pwd)/temp_bin:$PATH"
echo "DEBUG: PATH after buck2 override: $PATH"
echo "DEBUG: Running install_requirements.sh..."
./install_requirements.sh
echo "DEBUG: Finished install_requirements.sh."
echo "DEBUG: Removing temp_bin..."
rm -rf ./temp_bin
echo "DEBUG: Removed temp_bin."
