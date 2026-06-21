# Upgrade/rollback failure detection in bootc

This document describes how to detect when a reboot failed to enable the staged image in bootc.

## Overview

bootc uses different mechanisms to detect boot failures depending on the backend (OSTree vs. composefs+UKI) and the specific point of failure. Understanding these mechanisms is crucial for system administrators and automated tooling that needs to detect failed updates.

## OSTree Backend Boot Failure Detection

For systems using the traditional OSTree backend, bootc relies on OSTree's built-in boot failure detection mechanisms.

### Key Services

1. **`ostree-finalize-staged.service`** - Runs during shutdown to finalize staged deployments
2. **`ostree-boot-complete.service`** - Runs early in boot to detect finalization failures

When `ostree-finalize-staged.service` fails during shutdown/reboot, this will create
a stamp file in `/boot`, and then on a subsequent reboot the `ostree-boot-complete.service`
service will detect it, and then itself exit with a failure mode.

You can monitor the success of both services, though for `ostree-finalize-staged.service`
note that the failure occurred during the previous boot's shutdown.


## Composefs Backend Boot Failure Detection

### Key Services

There is a `bootc-finalize-staged.service` which is similar to `ostree-finalize-staged.service`,
but there is not currently a similar `-boot-complete.service`. There is also a `bootc-root-setup.service`
that runs during initramfs to mount the composefs image and set up `/etc` and `/var` - but if this
service fails, the system will not boot at all (emergency mode or hang).

At the current time then, it is recommended to check the journal for failures from the previous boot:

```bash
# Check for finalization failures from previous boot
journalctl -u bootc-finalize-staged.service -b -1
```

### Automatic rollback via systemd-boot boot counting

The composefs backend supports [systemd Automatic Boot Assessment](https://systemd.io/AUTOMATIC_BOOT_ASSESSMENT/)
(boot counting) on **systemd-boot**, mirroring what libostree provides for the OSTree backend
via [this commit](https://github.com/ostreedev/ostree/commit/08487091256b93493f8d692e37ab3d892c758da1).
When enabled, a newly-deployed image that repeatedly fails to boot is automatically
deprioritized by systemd-boot, so the previous (known-good) deployment boots instead — with no
manual `bootc rollback`.

#### Enabling

Boot counting is configured with systemd's standard `kernel-install` knob: a file at
`/etc/kernel/tries` in the image containing a single integer (the initial number of boot
attempts). A value of `3` is typical; an absent/empty file or `0` disables counting.

```dockerfile
# In your Containerfile (systemd-boot images only)
RUN echo 3 > /etc/kernel/tries
```

The value is read from the target image, so it applies on both `bootc install` and every
`bootc upgrade`.

#### How it works

When enabled, bootc writes the new/primary deployment's BLS entry with a tries counter in the
filename, e.g. `bootc_fedora-42.0-1+3.conf`. The currently-booted entry (the rollback target)
is never counted. systemd-boot decrements the counter on each boot attempt (renaming the file
on the ESP, e.g. `+3` → `+2-1`); when it reaches zero the entry is treated as "bad", sorted
last, and the previous deployment is booted instead.

> Note: boot counting is **systemd-boot only**. GRUB uses a separate `grubenv`-based mechanism
> (see [greenboot](https://github.com/fedora-iot/greenboot)) and is intentionally left
> untouched. The composefs UKI path is not yet wired for boot counting.

#### Required image-side configuration

bootc only writes the counter. For the loop to close, the **image** is responsible for marking
boots good — exactly as in the OSTree/greenboot world. Two pieces are required:

1. **Enable `systemd-bless-boot.service`** (ships with systemd). On a healthy boot it drops the
   counter from the filename (e.g. `bootc_fedora-42.0-1+2-1.conf` → `bootc_fedora-42.0-1.conf`),
   making the deployment permanently good.
2. **Gate `boot-complete.target`** on a health check, so it is only reached on a healthy boot.
   If `boot-complete.target` is never reached, `systemd-bless-boot.service` does not run, the
   decremented counter persists, and the deployment is eventually rolled back.

A minimal example health gate (provided by the image, not by bootc):

```ini
# /usr/lib/systemd/system/my-healthcheck.service
[Unit]
Description=Post-boot health check
DefaultDependencies=no
Before=boot-complete.target
Requisite=boot-complete.target

[Service]
Type=oneshot
ExecStart=/usr/libexec/my-healthcheck

[Install]
RequiredBy=boot-complete.target
```

[greenboot](https://github.com/fedora-iot/greenboot) is a batteries-included alternative that
wires `boot-complete.target` to a directory of health-check scripts.

> Verify the exact ordering/dependencies against your image's systemd version with
> `systemctl cat systemd-bless-boot.service boot-complete.target`.

## See Also

- [systemd Automatic Boot Assessment](https://systemd.io/AUTOMATIC_BOOT_ASSESSMENT/)
- [OSTree Manual](https://ostreedev.github.io/ostree/)
- [bootc-rollback(8)](man/bootc-rollback.8.md)
- [bootc-status(8)](man/bootc-status.8.md)
