use std::time::Duration;

use warp_multi_agent_api as api;

use super::{
    AIAgentActionResultType, LrcActivity, LrcFileActivity, LrcProcessActivity, LrcProcessState,
    RunAgentsAgentOutcome, RunAgentsAgentOutcomeKind, RunAgentsLaunchedExecutionMode,
    RunAgentsResult,
};

fn launched_agent(name: &str) -> RunAgentsAgentOutcome {
    RunAgentsAgentOutcome {
        name: name.to_string(),
        kind: RunAgentsAgentOutcomeKind::Launched {
            agent_id: format!("{name}-id"),
        },
        resolved_model_id: String::new(),
    }
}

fn failed_agent(name: &str) -> RunAgentsAgentOutcome {
    RunAgentsAgentOutcome {
        name: name.to_string(),
        kind: RunAgentsAgentOutcomeKind::Failed {
            error: "launch failed".to_string(),
        },
        resolved_model_id: String::new(),
    }
}

fn run_agents_result(agents: Vec<RunAgentsAgentOutcome>) -> AIAgentActionResultType {
    AIAgentActionResultType::RunAgents(RunAgentsResult::Launched {
        model_id: "auto".to_string(),
        harness_type: "oz".to_string(),
        execution_mode: RunAgentsLaunchedExecutionMode::Local,
        agents,
    })
}

#[test]
fn run_agents_is_successful_when_all_agents_launch() {
    let result = run_agents_result(vec![launched_agent("first"), launched_agent("second")]);

    assert!(result.is_successful());
    assert!(!result.is_failed());
}

#[test]
fn run_agents_is_successful_when_some_agents_launch() {
    let result = run_agents_result(vec![launched_agent("first"), failed_agent("second")]);
    assert!(result.is_successful());
    assert!(!result.is_failed());
}

#[test]
fn run_agents_is_failed_when_no_agents_launch() {
    let result = run_agents_result(vec![failed_agent("first"), failed_agent("second")]);

    assert!(!result.is_successful());
    assert!(result.is_failed());
}

fn populated_activity() -> LrcActivity {
    LrcActivity {
        since_last_activity: Some(Duration::from_millis(1500)),
        output_changed_since_last_read: true,
        since_output_change: Some(Duration::from_secs(42)),
        process: Some(LrcProcessActivity {
            cpu_time_delta: Duration::from_millis(2750),
            state: LrcProcessState::DiskWait,
            live_process_count: 3,
            io_write_bytes_delta: 4096,
        }),
        files: vec![LrcFileActivity {
            path: "/tmp/build.log".to_string(),
            size_bytes: 8192,
            size_delta_bytes: -128,
            tail: "linking\n".to_string(),
        }],
        signals_unavailable: false,
    }
}

#[test]
fn activity_survives_a_round_trip_through_the_api_type() {
    let activity = populated_activity();

    let wire = api::LongRunningCommandActivity::from(activity.clone());
    assert_eq!(LrcActivity::from(&wire), activity);
}

#[test]
fn activity_converts_durations_to_seconds_on_the_wire() {
    let wire = api::LongRunningCommandActivity::from(populated_activity());

    assert_eq!(wire.seconds_since_last_activity, 1.5);
    assert_eq!(wire.seconds_since_output_change, 42.0);
    assert_eq!(wire.process.expect("process tier").cpu_time_delta_ms, 2750);
}

#[test]
fn a_tier_that_never_reported_activity_is_sent_as_zero_seconds() {
    let wire = api::LongRunningCommandActivity::from(LrcActivity::default());

    assert_eq!(wire.seconds_since_last_activity, 0.0);
    assert_eq!(wire.seconds_since_output_change, 0.0);
    assert!(wire.process.is_none());
    assert!(wire.files.is_empty());
}

#[test]
fn unavailable_signals_survive_the_round_trip() {
    let activity = LrcActivity {
        signals_unavailable: true,
        ..Default::default()
    };

    let wire = api::LongRunningCommandActivity::from(activity);

    assert!(wire.signals_unavailable);
    assert!(LrcActivity::from(&wire).signals_unavailable);
}

/// The wire type has no way to say "never", so a tier that never reported
/// activity is indistinguishable from one that reported it just now. The
/// tier-level fields alongside it are what tell the two apart.
#[test]
fn a_never_reported_tier_reads_back_as_zero_rather_than_never() {
    let wire = api::LongRunningCommandActivity::from(LrcActivity::default());

    assert_eq!(
        LrcActivity::from(&wire).since_last_activity,
        Some(Duration::ZERO)
    );
}

#[test]
fn an_unrecognized_process_state_reads_back_as_unknown() {
    let wire = api::long_running_command_activity::ProcessActivity {
        state: "something-new".to_string(),
        ..Default::default()
    };

    assert_eq!(
        LrcProcessActivity::from(&wire).state,
        LrcProcessState::Unknown
    );
}
