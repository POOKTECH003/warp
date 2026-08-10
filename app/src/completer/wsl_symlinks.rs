use std::collections::{HashMap, HashSet};
use std::fs::ReadDir;
use std::path::Path;
use std::time::Duration;

use anyhow::anyhow;
use instant::Instant;
use typed_path::TypedPath;
use warp_completer::completer::{CommandExitStatus, EngineDirEntry, EngineFileType};
use warp_util::path::ShellFamily;
use warpui::r#async::FutureExt as AsyncFutureExt;

use crate::completer::SessionContext;
use crate::terminal::model::session::ExecuteCommandOptions;

const GUEST_CLASSIFY_TIMEOUT: Duration = Duration::from_secs(3);

/// The Windows host cannot resolve a symlink whose target lives in the WSL guest path space, so it
/// classifies such a directory symlink as a file and the guest has to correct it.
pub(super) async fn list_entries(
    session_context: &SessionContext,
    directory: &TypedPath<'_>,
    read_dir: ReadDir,
) -> Vec<EngineDirEntry> {
    let mut entries = Vec::new();
    let mut unresolved_symlinks: HashSet<String> = HashSet::new();
    for entry in read_dir.filter_map(|res| res.ok()) {
        let is_symlink = entry
            .file_type()
            .map(|file_type| file_type.is_symlink())
            .unwrap_or(false);
        let Ok(engine_entry) = EngineDirEntry::try_from(entry) else {
            continue;
        };
        if is_symlink && !engine_entry.is_dir() {
            unresolved_symlinks.insert(engine_entry.file_name().to_owned());
        }
        entries.push(engine_entry);
    }

    if !unresolved_symlinks.is_empty() && session_context.session.is_wsl() {
        let started = Instant::now();
        let outcome = classify_with_guest(session_context, directory).await;
        let elapsed_ms = started.elapsed().as_millis();
        let unresolved = unresolved_symlinks.len();
        match outcome {
            GuestOutcome::Classified(guest_dirs) => {
                let upgraded =
                    upgrade_directory_symlinks(&mut entries, &unresolved_symlinks, &guest_dirs);
                log::debug!(
                    "[APP-3993 wsl-classify] ok upgraded={upgraded} unresolved={unresolved} elapsed_ms={elapsed_ms}"
                );
            }
            GuestOutcome::NonZeroExit => log::warn!(
                "[APP-3993 wsl-classify] non-zero exit unresolved={unresolved} elapsed_ms={elapsed_ms}"
            ),
            GuestOutcome::Failed(err) => log::warn!(
                "[APP-3993 wsl-classify] failed unresolved={unresolved} elapsed_ms={elapsed_ms}: {err:#}"
            ),
            GuestOutcome::TimedOut => log::warn!(
                "[APP-3993 wsl-classify] timed out unresolved={unresolved} elapsed_ms={elapsed_ms}"
            ),
        }
    }

    entries
}

/// Keeps the guest failure modes distinct so one call site can report the outcome with its timing.
enum GuestOutcome {
    Classified(HashSet<String>),
    NonZeroExit,
    Failed(anyhow::Error),
    TimedOut,
}

/// Asks the guest which immediate children of `directory` are directories. Any failure leaves the
/// host classification in place rather than stalling completion on a slow guest.
async fn classify_with_guest(
    session_context: &SessionContext,
    directory: &TypedPath<'_>,
) -> GuestOutcome {
    let Some(script) = find_dirs_script_for_dir(directory) else {
        return GuestOutcome::Failed(anyhow!("directory path is not valid UTF-8"));
    };
    let env_vars = session_context
        .session
        .path()
        .as_deref()
        .map(|path| HashMap::from_iter([("PATH".to_string(), path.to_string())]));
    let result = session_context
        .session
        .execute_command(&script, None, env_vars, ExecuteCommandOptions::default())
        .with_timeout(GUEST_CLASSIFY_TIMEOUT)
        .await;
    match result {
        Ok(Ok(output)) if output.status == CommandExitStatus::Success => match output.to_string() {
            Ok(output) => GuestOutcome::Classified(parse_directory_names(&output)),
            Err(err) => GuestOutcome::Failed(err),
        },
        Ok(Ok(_)) => GuestOutcome::NonZeroExit,
        Ok(Err(err)) => GuestOutcome::Failed(err),
        Err(_timed_out) => GuestOutcome::TimedOut,
    }
}

/// Follows symlinks (`-L`) so a directory symlink the host cannot resolve is still reported. Only
/// the directory is interpolated, so the listing contributes no quoting or injection surface.
fn find_dirs_script_for_dir(directory: &TypedPath) -> Option<String> {
    let dir_str = directory.to_str()?;
    let escaped_dir = ShellFamily::Posix.shell_escape(dir_str);
    Some(format!("cd {escaped_dir} && find -L . -maxdepth 1 -type d -print0").replace('\n', " "))
}

/// Drops the listed directory itself, which `find` emits as ".".
fn parse_directory_names(output: &str) -> HashSet<String> {
    output
        .split('\0')
        .filter(|entry| !entry.is_empty() && *entry != ".")
        .filter_map(|entry| Path::new(entry).file_name().and_then(|name| name.to_str()))
        .map(|name| name.to_owned())
        .collect()
}

/// Only entries the host left as unresolved symlinks are upgraded, so a non-symlink the guest
/// happens to list as a directory keeps the host's classification. Returns how many were upgraded.
fn upgrade_directory_symlinks(
    entries: &mut [EngineDirEntry],
    unresolved_symlinks: &HashSet<String>,
    guest_dirs: &HashSet<String>,
) -> usize {
    let mut upgraded = 0;
    for entry in entries.iter_mut() {
        if !entry.is_dir()
            && unresolved_symlinks.contains(entry.file_name())
            && guest_dirs.contains(entry.file_name())
        {
            entry.file_type = EngineFileType::Directory;
            upgraded += 1;
        }
    }
    upgraded
}

#[cfg(test)]
#[path = "wsl_symlinks_tests.rs"]
mod tests;
