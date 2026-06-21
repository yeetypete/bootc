//! systemd-boot Automatic Boot Assessment (boot counting) for the composefs backend.
//!
//! When enabled, bootc appends a "tries" counter (`+N`) to the newly-deployed entry's BLS
//! `.conf` filename on systemd-boot. systemd-boot decrements the counter on each boot attempt
//! (renaming the file on the ESP), and a deployment that never reaches `boot-complete.target`
//! is eventually deprioritized so the previous, known-good deployment boots instead. This
//! mirrors what libostree already does for the ostree backend via `boot-counting-tries`; see
//! <https://systemd.io/AUTOMATIC_BOOT_ASSESSMENT/> and the Boot Loader Specification.
//!
//! bootc only writes the counter. For the loop to close, the **image** must:
//! 1. enable the stock `systemd-bless-boot.service` (ships with systemd), which drops the
//!    counter on a healthy boot, and
//! 2. gate `boot-complete.target` on a health check (e.g. greenboot, or a custom unit).
//!
//! bootc deliberately ships none of those units, matching ostree/libostree.
//!
//! Configuration uses systemd's standard `/etc/kernel/tries` (`kernel-install`'s convention,
//! which ostree also honors): a file containing a single integer read from the target image.
//! Absent / empty / `0` disables counting.

use anyhow::{Context, Result};
use cap_std_ext::cap_std::fs::Dir;
use cap_std_ext::dirext::CapStdExtDirExt;

/// systemd's standard location for the initial boot-counter value (`kernel-install`).
const KERNEL_TRIES_PATH: &str = "etc/kernel/tries";

/// Returns `Some(tries)` if boot counting is enabled in `target_root`, else `None`.
///
/// `target_root` is the root of the image being deployed (the mounted EROFS), so the policy
/// is honored uniformly on both install and every upgrade. The value is read from systemd's
/// standard `/etc/kernel/tries`. A missing file, an empty file, or a value of `0` all disable
/// counting. Any positive value is returned as-is.
pub(crate) fn boot_counting_tries(target_root: &Dir) -> Result<Option<u32>> {
    let Some(contents) = target_root
        .read_to_string_optional(KERNEL_TRIES_PATH)
        .with_context(|| format!("Reading {KERNEL_TRIES_PATH}"))?
    else {
        return Ok(None);
    };

    let trimmed = contents.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let tries: u32 = trimmed
        .parse()
        .with_context(|| format!("Parsing {KERNEL_TRIES_PATH}: invalid integer {trimmed:?}"))?;

    // A value of 0 means "disabled" (a `+0` entry would boot straight to "bad").
    Ok((tries > 0).then_some(tries))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cap_std_ext::cap_std::ambient_authority;
    use cap_std_ext::cap_tempfile::TempDir;

    fn tempdir() -> TempDir {
        cap_std_ext::cap_tempfile::tempdir(ambient_authority()).unwrap()
    }

    fn write_tries(dir: &Dir, contents: &str) {
        dir.create_dir_all("etc/kernel").unwrap();
        dir.write(KERNEL_TRIES_PATH, contents).unwrap();
    }

    #[test]
    fn missing_file_disables() {
        let d = tempdir();
        assert_eq!(boot_counting_tries(&d).unwrap(), None);
    }

    #[test]
    fn positive_value_enables() {
        let d = tempdir();
        write_tries(&d, "3");
        assert_eq!(boot_counting_tries(&d).unwrap(), Some(3));
    }

    #[test]
    fn trailing_newline_ok() {
        let d = tempdir();
        write_tries(&d, "3\n");
        assert_eq!(boot_counting_tries(&d).unwrap(), Some(3));
    }

    #[test]
    fn zero_disables() {
        let d = tempdir();
        write_tries(&d, "0");
        assert_eq!(boot_counting_tries(&d).unwrap(), None);
    }

    #[test]
    fn empty_disables() {
        let d = tempdir();
        write_tries(&d, "   \n");
        assert_eq!(boot_counting_tries(&d).unwrap(), None);
    }

    #[test]
    fn non_integer_errors() {
        let d = tempdir();
        write_tries(&d, "yes");
        assert!(boot_counting_tries(&d).is_err());
    }
}
