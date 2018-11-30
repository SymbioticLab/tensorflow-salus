FROM registry.gitlab.com/salus/builder

# Install ssh server and other development tools
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && (echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections) \
    && apt-get install -y openssh-server rsync

# Config the sshd server, the root password is 'root'.
# For development use ONLY, NEVER expose this to the Internet!!!
RUN mkdir /var/run/sshd \
    && echo 'root:root' | chpasswd \
    && sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config \
    && mkdir /root/.ssh

RUN spack install gdb \
    && spack view -d false -v add packages gdb \
    && spack-pin

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
