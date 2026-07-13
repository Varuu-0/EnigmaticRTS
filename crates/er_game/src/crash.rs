//! Crash diagnostics written by the process-wide panic hook.

use std::{
    backtrace::Backtrace,
    fs, panic,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

pub fn install_crash_hook() {
    let default_hook = panic::take_hook();

    panic::set_hook(Box::new(move |panic_info| {
        let backtrace = Backtrace::force_capture();
        let location = panic_info
            .location()
            .map(|location| {
                format!(
                    "{}:{}:{}",
                    location.file(),
                    location.line(),
                    location.column()
                )
            })
            .unwrap_or_else(|| "<unknown location>".to_owned());
        let message = panic_message(panic_info);
        let timestamp = unix_timestamp();
        let report = format!(
            "EnigmaticRTS crash report\n\
             timestamp_unix: {timestamp}\n\
             location: {location}\n\
             message: {message}\n\n\
             Backtrace:\n{backtrace}\n"
        );

        eprintln!("{report}");
        if let Some(path) = write_report(timestamp, &report) {
            eprintln!("Crash report written to: {}", path.display());
        }

        default_hook(panic_info);
    }));
}

fn panic_message(panic_info: &panic::PanicHookInfo<'_>) -> String {
    if let Some(message) = panic_info.payload().downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = panic_info.payload().downcast_ref::<String>() {
        message.clone()
    } else {
        "<non-string panic payload>".to_owned()
    }
}

fn write_report(timestamp: u64, report: &str) -> Option<PathBuf> {
    let directory = crash_report_directory()?;
    fs::create_dir_all(&directory).ok()?;
    let path = directory.join(format!("crash_{timestamp}.txt"));
    fs::write(&path, report).ok()?;
    Some(path)
}

fn crash_report_directory() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("crash_reports")))
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|directory| directory.join("crash_reports"))
        })
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}
