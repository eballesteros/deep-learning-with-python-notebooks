
FROM tensorflow/tensorflow:2.6.1-gpu-jupyter

# some other installs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install scikit-learn==1.0.2 \
    tensorflow-hub==0.12.0 tensorflow-datasets==4.5.2 \
    seaborn==0.11.2 \
    # Install pyyaml and h5py for saving model.
    pyyaml==6.0 h5py==3.6.0 \
    # Debung Jupiter notebook on Visual Studio Code.
    ipykernel==6.9.0 \
    # For `tensorflow.keras.utils.plot_model`
    pydot==1.4.2

# create non-root user
# https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user
ARG USERNAME=nonroot
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user 
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
    #
    # # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    # && apt-get update \
    # && apt-get install -y sudo \
    # && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    # && chmod 0440 /etc/sudoers.d/$USERNAME

#  We want to run common-debian.sh from here:
# https://github.com/microsoft/vscode-dev-containers/tree/main/script-library#development-container-scripts
COPY library-scripts/common-debian.sh /tmp/library-scripts/

RUN apt-get update \
    && bash /tmp/library-scripts/common-debian.sh \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME