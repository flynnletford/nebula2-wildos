FROM ros:humble-ros-core-jammy

# Install build essentials and development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    ros-humble-rosidl-default-generators \
    ros-humble-rosidl-default-runtime \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS2 workspace
WORKDIR /ros2_ws

# Source ROS2 setup in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

# Default entrypoint to bash shell
ENTRYPOINT ["/bin/bash"]
CMD ["-i"]
