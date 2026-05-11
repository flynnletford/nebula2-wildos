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
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS2 workspace
WORKDIR /ros2_ws

# Copy model checkpoints into the container
COPY model_ckpts /ros2_ws/model_ckpts

# Copy ROS2 source code
COPY src /ros2_ws/src

# Build ROS2 packages with colcon
RUN . /opt/ros/humble/setup.sh && colcon build

# Source ROS2 and built packages in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /etc/bash.bashrc

# Default entrypoint to bash shell
ENTRYPOINT ["/bin/bash"]
CMD ["-i"]
