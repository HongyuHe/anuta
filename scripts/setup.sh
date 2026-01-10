#!/usr/bin/env bash
set -x

{
    #* Check if /nfs is mounted; if not, run the client setup.
    if mountpoint -q /nfs; then
        echo "/nfs is already mounted."
    else
        echo "/nfs is not mounted. Mounting now..."
        sudo /bin/bash /local/repository/nfs-client.sh
    fi

    sudo apt update -y && sudo apt upgrade -y

    # * Update python to 3.10
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get install python3.10 -y
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2 # * Larger number, higher priority
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2

    # * Install pip with the right version.
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
    sudo cp /users/"$USER"/.local/bin/pip /usr/bin/
    sudo apt install python3.10-distutils -y

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

    sudo apt install htop -y
    git config --global credential.helper store

    # unzip data/meta.zip -d data/
    bash scripts/install_java.sh

    echo "set -g history-limit 10000000" > ~/.tmux.conf
    tmux source-file ~/.tmux.conf

    exit 0
}
