// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gateway-owned compute orchestration over a pluggable compute backend.

pub mod vm;

pub use vm::VmComputeConfig;

use crate::grpc::policy::{SANDBOX_SETTINGS_OBJECT_TYPE, sandbox_settings_id};
use crate::persistence::{ObjectId, ObjectName, ObjectRecord, ObjectType, Store};
use crate::sandbox_index::SandboxIndex;
use crate::sandbox_watch::SandboxWatchBus;
use crate::tracing_bus::TracingLogBus;
use futures::{Stream, StreamExt};
use openshell_core::proto::compute::v1::{
    CreateSandboxRequest, DeleteSandboxRequest, DriverCondition, DriverPlatformEvent,
    DriverResourceRequirements, DriverSandbox, DriverSandboxSpec, DriverSandboxStatus,
    DriverSandboxTemplate, GetCapabilitiesRequest, GetSandboxRequest, ListSandboxesRequest,
    ValidateSandboxCreateRequest, WatchSandboxesEvent, WatchSandboxesRequest,
    compute_driver_client::ComputeDriverClient, compute_driver_server::ComputeDriver,
    watch_sandboxes_event,
};
use openshell_core::proto::{
    PlatformEvent, Sandbox, SandboxCondition, SandboxPhase, SandboxSpec, SandboxStatus,
    SandboxTemplate, SshSession,
};
use openshell_driver_kubernetes::{
    ComputeDriverService, KubernetesComputeConfig, KubernetesComputeDriver,
};
use prost::Message;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tonic::transport::Channel;
use tonic::{Code, Request, Status};
use tracing::{info, warn};

type DriverWatchStream = Pin<Box<dyn Stream<Item = Result<WatchSandboxesEvent, Status>> + Send>>;
type SharedComputeDriver =
    Arc<dyn ComputeDriver<WatchSandboxesStream = DriverWatchStream> + Send + Sync>;

/// Interval between store-vs-backend reconciliation sweeps.
const RECONCILE_INTERVAL: Duration = Duration::from_secs(60);

/// How long a sandbox can remain provisioning in the store without a
/// corresponding backend resource before it is considered orphaned.
const ORPHAN_GRACE_PERIOD: Duration = Duration::from_secs(300);

#[derive(Debug, thiserror::Error)]
pub enum ComputeError {
    #[error("sandbox already exists")]
    AlreadyExists,
    #[error("{0}")]
    Precondition(String),
    #[error("{0}")]
    Message(String),
}
#[derive(Debug)]
pub(crate) struct ManagedDriverProcess {
    child: std::sync::Mutex<Option<tokio::process::Child>>,
    socket_path: std::path::PathBuf,
}

impl ManagedDriverProcess {
    pub(crate) fn new(child: tokio::process::Child, socket_path: std::path::PathBuf) -> Self {
        Self {
            child: std::sync::Mutex::new(Some(child)),
            socket_path,
        }
    }
}

impl Drop for ManagedDriverProcess {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.take();
        }
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

#[derive(Debug, Clone)]
struct RemoteComputeDriver {
    channel: Channel,
}

impl RemoteComputeDriver {
    fn new(channel: Channel) -> Self {
        Self { channel }
    }

    fn client(&self) -> ComputeDriverClient<Channel> {
        ComputeDriverClient::new(self.channel.clone())
    }
}

#[tonic::async_trait]
impl ComputeDriver for RemoteComputeDriver {
    type WatchSandboxesStream = DriverWatchStream;

    async fn get_capabilities(
        &self,
        request: Request<GetCapabilitiesRequest>,
    ) -> Result<tonic::Response<openshell_core::proto::compute::v1::GetCapabilitiesResponse>, Status>
    {
        let mut client = self.client();
        client.get_capabilities(request).await
    }

    async fn validate_sandbox_create(
        &self,
        request: Request<ValidateSandboxCreateRequest>,
    ) -> Result<
        tonic::Response<openshell_core::proto::compute::v1::ValidateSandboxCreateResponse>,
        Status,
    > {
        let mut client = self.client();
        client.validate_sandbox_create(request).await
    }

    async fn get_sandbox(
        &self,
        request: Request<GetSandboxRequest>,
    ) -> Result<tonic::Response<openshell_core::proto::compute::v1::GetSandboxResponse>, Status>
    {
        let mut client = self.client();
        client.get_sandbox(request).await
    }

    async fn list_sandboxes(
        &self,
        request: Request<ListSandboxesRequest>,
    ) -> Result<tonic::Response<openshell_core::proto::compute::v1::ListSandboxesResponse>, Status>
    {
        let mut client = self.client();
        client.list_sandboxes(request).await
    }

    async fn create_sandbox(
        &self,
        request: Request<CreateSandboxRequest>,
    ) -> Result<tonic::Response<openshell_core::proto::compute::v1::CreateSandboxResponse>, Status>
    {
        let mut client = self.client();
        client.create_sandbox(request).await
    }

    async fn stop_sandbox(
        &self,
        request: Request<openshell_core::proto::compute::v1::StopSandboxRequest>,
    ) -> Result<tonic::Response<openshell_core::proto::compute::v1::StopSandboxResponse>, Status>
    {
        let mut client = self.client();
        client.stop_sandbox(request).await
    }

    async fn delete_sandbox(
        &self,
        request: Request<DeleteSandboxRequest>,
    ) -> Result<tonic::Response<openshell_core::proto::compute::v1::DeleteSandboxResponse>, Status>
    {
        let mut client = self.client();
        client.delete_sandbox(request).await
    }

    async fn watch_sandboxes(
        &self,
        request: Request<WatchSandboxesRequest>,
    ) -> Result<tonic::Response<Self::WatchSandboxesStream>, Status> {
        let mut client = self.client();
        let response = client.watch_sandboxes(request).await?;
        let stream = response
            .into_inner()
            .map(|item| item.map_err(|status| status));
        Ok(tonic::Response::new(Box::pin(stream)))
    }
}

#[derive(Clone)]
pub struct ComputeRuntime {
    driver: SharedComputeDriver,
    _driver_process: Option<Arc<ManagedDriverProcess>>,
    default_image: String,
    store: Arc<Store>,
    sandbox_index: SandboxIndex,
    sandbox_watch_bus: SandboxWatchBus,
    tracing_log_bus: TracingLogBus,
    sync_lock: Arc<Mutex<()>>,
    /// Optional weak handle to the supervisor session registry, installed
    /// via [`Self::install_supervisor_observer`] during server startup.
    /// Used by [`Self::apply_sandbox_update_locked`] to backfill Ready when
    /// the supervisor session registered before the driver snapshot arrived.
    ///
    /// Held as [`std::sync::Weak`] to break the cycle between the
    /// registry (owned by `ServerState` via `Arc`) and the observer it
    /// installs (which owns a cloned [`ComputeRuntime`]). Without
    /// `Weak` here, dropping `ServerState` would not free either side.
    /// Stored inside a [`std::sync::Mutex`] because the access pattern
    /// is a single write during startup and cheap reads after;
    /// `tokio::sync::Mutex` would force awaits in otherwise-sync paths.
    supervisor_sessions: Arc<
        std::sync::Mutex<
            Option<std::sync::Weak<crate::supervisor_session::SupervisorSessionRegistry>>,
        >,
    >,
}

impl fmt::Debug for ComputeRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputeRuntime").finish_non_exhaustive()
    }
}

impl ComputeRuntime {
    async fn from_driver(
        driver: SharedComputeDriver,
        driver_process: Option<Arc<ManagedDriverProcess>>,
        store: Arc<Store>,
        sandbox_index: SandboxIndex,
        sandbox_watch_bus: SandboxWatchBus,
        tracing_log_bus: TracingLogBus,
    ) -> Result<Self, ComputeError> {
        let default_image = driver
            .get_capabilities(Request::new(GetCapabilitiesRequest {}))
            .await
            .map_err(compute_error_from_status)?
            .into_inner()
            .default_image;
        Ok(Self {
            driver,
            _driver_process: driver_process,
            default_image,
            store,
            sandbox_index,
            sandbox_watch_bus,
            tracing_log_bus,
            sync_lock: Arc::new(Mutex::new(())),
            supervisor_sessions: Arc::new(std::sync::Mutex::new(None)),
        })
    }

    /// Bind the runtime to the server's supervisor session registry. Called
    /// once during server startup, after both objects exist. Enables:
    ///
    /// 1. The registry's `SupervisorSessionObserver` implementation below to
    ///    fire [`Self::mark_sandbox_session_connected`] /
    ///    [`Self::mark_sandbox_session_disconnected`] on session lifecycle
    ///    events.
    /// 2. [`Self::apply_sandbox_update_locked`] to consult
    ///    [`crate::supervisor_session::SupervisorSessionRegistry::has_session`]
    ///    and backfill Ready when the supervisor already connected before
    ///    the driver reported the sandbox.
    pub fn install_supervisor_observer(
        &self,
        registry: &Arc<crate::supervisor_session::SupervisorSessionRegistry>,
    ) {
        *self.supervisor_sessions.lock().unwrap() = Some(Arc::downgrade(registry));
        registry.set_observer(Arc::new(ComputeSessionObserver {
            compute: self.clone(),
        }));
    }

    /// Cheap snapshot of the supervisor registry handle. Returns `None`
    /// when [`Self::install_supervisor_observer`] was never called, or
    /// when the underlying registry has been dropped (test teardown).
    fn supervisor_registry(
        &self,
    ) -> Option<Arc<crate::supervisor_session::SupervisorSessionRegistry>> {
        self.supervisor_sessions
            .lock()
            .unwrap()
            .as_ref()
            .and_then(std::sync::Weak::upgrade)
    }

    pub async fn new_kubernetes(
        config: KubernetesComputeConfig,
        store: Arc<Store>,
        sandbox_index: SandboxIndex,
        sandbox_watch_bus: SandboxWatchBus,
        tracing_log_bus: TracingLogBus,
    ) -> Result<Self, ComputeError> {
        let driver = KubernetesComputeDriver::new(config)
            .await
            .map_err(|err| ComputeError::Message(err.to_string()))?;
        let driver: SharedComputeDriver = Arc::new(ComputeDriverService::new(driver));
        Self::from_driver(
            driver,
            None,
            store,
            sandbox_index,
            sandbox_watch_bus,
            tracing_log_bus,
        )
        .await
    }

    pub(crate) async fn new_remote_vm(
        channel: Channel,
        driver_process: Option<Arc<ManagedDriverProcess>>,
        store: Arc<Store>,
        sandbox_index: SandboxIndex,
        sandbox_watch_bus: SandboxWatchBus,
        tracing_log_bus: TracingLogBus,
    ) -> Result<Self, ComputeError> {
        let driver: SharedComputeDriver = Arc::new(RemoteComputeDriver::new(channel));
        Self::from_driver(
            driver,
            driver_process,
            store,
            sandbox_index,
            sandbox_watch_bus,
            tracing_log_bus,
        )
        .await
    }

    #[must_use]
    pub fn default_image(&self) -> &str {
        &self.default_image
    }

    pub async fn validate_sandbox_create(&self, sandbox: &Sandbox) -> Result<(), Status> {
        let driver_sandbox = driver_sandbox_from_public(sandbox);
        self.driver
            .validate_sandbox_create(Request::new(ValidateSandboxCreateRequest {
                sandbox: Some(driver_sandbox),
            }))
            .await
            .map(|_| ())
    }

    pub async fn create_sandbox(&self, sandbox: Sandbox) -> Result<Sandbox, Status> {
        let existing = self
            .store
            .get_message_by_name::<Sandbox>(&sandbox.name)
            .await
            .map_err(|e| Status::internal(format!("fetch sandbox failed: {e}")))?;
        if existing.is_some() {
            return Err(Status::already_exists(format!(
                "sandbox '{}' already exists",
                sandbox.name
            )));
        }

        self.sandbox_index.update_from_sandbox(&sandbox);
        self.store
            .put_message(&sandbox)
            .await
            .map_err(|e| Status::internal(format!("persist sandbox failed: {e}")))?;

        let driver_sandbox = driver_sandbox_from_public(&sandbox);
        match self
            .driver
            .create_sandbox(Request::new(CreateSandboxRequest {
                sandbox: Some(driver_sandbox),
            }))
            .await
        {
            Ok(_) => {
                self.sandbox_watch_bus.notify(&sandbox.id);
                Ok(sandbox)
            }
            Err(status) if status.code() == Code::AlreadyExists => {
                let _ = self.store.delete(Sandbox::object_type(), &sandbox.id).await;
                self.sandbox_index.remove_sandbox(&sandbox.id);
                Err(Status::already_exists("sandbox already exists"))
            }
            Err(status) if status.code() == Code::FailedPrecondition => {
                let _ = self.store.delete(Sandbox::object_type(), &sandbox.id).await;
                self.sandbox_index.remove_sandbox(&sandbox.id);
                Err(Status::failed_precondition(status.message().to_string()))
            }
            Err(err) => {
                let _ = self.store.delete(Sandbox::object_type(), &sandbox.id).await;
                self.sandbox_index.remove_sandbox(&sandbox.id);
                Err(Status::internal(format!(
                    "create sandbox failed: {}",
                    err.message()
                )))
            }
        }
    }

    pub async fn delete_sandbox(&self, name: &str) -> Result<bool, Status> {
        let sandbox = self
            .store
            .get_message_by_name::<Sandbox>(name)
            .await
            .map_err(|e| Status::internal(format!("fetch sandbox failed: {e}")))?;

        let Some(mut sandbox) = sandbox else {
            return Err(Status::not_found("sandbox not found"));
        };

        let id = sandbox.id.clone();
        sandbox.phase = SandboxPhase::Deleting as i32;
        self.store
            .put_message(&sandbox)
            .await
            .map_err(|e| Status::internal(format!("persist sandbox failed: {e}")))?;
        self.sandbox_index.update_from_sandbox(&sandbox);
        self.sandbox_watch_bus.notify(&id);

        if let Ok(records) = self.store.list(SshSession::object_type(), 1000, 0).await {
            for record in records {
                if let Ok(session) = SshSession::decode(record.payload.as_slice())
                    && session.sandbox_id == id
                    && let Err(e) = self
                        .store
                        .delete(SshSession::object_type(), &session.id)
                        .await
                {
                    warn!(
                        session_id = %session.id,
                        error = %e,
                        "Failed to delete SSH session during sandbox cleanup"
                    );
                }
            }
        }

        if let Err(e) = self
            .store
            .delete(SANDBOX_SETTINGS_OBJECT_TYPE, &sandbox_settings_id(&id))
            .await
        {
            warn!(
                sandbox_id = %id,
                error = %e,
                "Failed to delete sandbox settings during cleanup"
            );
        }

        let driver_sandbox = driver_sandbox_from_public(&sandbox);
        let deleted = self
            .driver
            .delete_sandbox(Request::new(DeleteSandboxRequest {
                sandbox_id: driver_sandbox.id,
                sandbox_name: driver_sandbox.name,
            }))
            .await
            .map(|response| response.into_inner().deleted)
            .map_err(|err| Status::internal(format!("delete sandbox failed: {}", err.message())))?;

        if !deleted && let Err(e) = self.store.delete(Sandbox::object_type(), &id).await {
            warn!(sandbox_id = %id, error = %e, "Failed to clean up store after delete");
        }

        self.cleanup_sandbox_state(&id);
        Ok(deleted)
    }

    pub fn spawn_watchers(&self) {
        let runtime = Arc::new(self.clone());
        let watch_runtime = runtime.clone();
        tokio::spawn(async move {
            watch_runtime.watch_loop().await;
        });
        tokio::spawn(async move {
            runtime.reconcile_loop().await;
        });
    }

    async fn watch_loop(self: Arc<Self>) {
        loop {
            let mut stream = match self
                .driver
                .watch_sandboxes(Request::new(WatchSandboxesRequest {}))
                .await
            {
                Ok(response) => response.into_inner(),
                Err(err) => {
                    warn!(error = %err, "Compute driver watch stream failed to start");
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    continue;
                }
            };

            let mut restart = false;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(event) => {
                        if let Err(err) = self.apply_watch_event(event).await {
                            warn!(error = %err, "Failed to apply compute driver event");
                        }
                    }
                    Err(err) => {
                        warn!(error = %err, "Compute driver watch stream errored");
                        restart = true;
                        break;
                    }
                }
            }

            if !restart {
                warn!("Compute driver watch stream ended unexpectedly");
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }

    async fn reconcile_loop(self: Arc<Self>) {
        loop {
            if let Err(err) = self.reconcile_store_with_backend(ORPHAN_GRACE_PERIOD).await {
                warn!(error = %err, "Store reconciliation sweep failed");
            }
            tokio::time::sleep(RECONCILE_INTERVAL).await;
        }
    }

    async fn reconcile_store_with_backend(&self, grace_period: Duration) -> Result<(), String> {
        let sweep_started_at_ms = current_time_ms();
        let backend_sandboxes = self
            .driver
            .list_sandboxes(Request::new(ListSandboxesRequest {}))
            .await
            .map_err(|e| e.to_string())?
            .into_inner()
            .sandboxes;
        let backend_ids = backend_sandboxes
            .iter()
            .map(|sandbox| sandbox.id.clone())
            .collect::<std::collections::HashSet<_>>();

        for sandbox in backend_sandboxes {
            self.reconcile_snapshot_sandbox(sandbox, sweep_started_at_ms)
                .await?;
        }

        let records = self
            .store
            .list(Sandbox::object_type(), 500, 0)
            .await
            .map_err(|e| e.to_string())?;

        let grace_ms = grace_period.as_millis().try_into().unwrap_or(i64::MAX);

        for record in records {
            let sandbox = match Sandbox::decode(record.payload.as_slice()) {
                Ok(sandbox) => sandbox,
                Err(err) => {
                    warn!(error = %err, "Failed to decode sandbox record during reconciliation");
                    continue;
                }
            };

            if backend_ids.contains(&sandbox.id) {
                continue;
            }

            self.prune_missing_sandbox(record, sweep_started_at_ms, grace_ms)
                .await?;
        }

        Ok(())
    }

    async fn apply_watch_event(&self, event: WatchSandboxesEvent) -> Result<(), String> {
        match event.payload {
            Some(watch_sandboxes_event::Payload::Sandbox(sandbox)) => {
                if let Some(sandbox) = sandbox.sandbox {
                    self.apply_sandbox_update(sandbox).await?;
                }
            }
            Some(watch_sandboxes_event::Payload::Deleted(deleted)) => {
                self.apply_deleted(&deleted.sandbox_id).await?;
            }
            Some(watch_sandboxes_event::Payload::PlatformEvent(platform_event)) => {
                if let Some(event) = platform_event.event {
                    self.tracing_log_bus.platform_event_bus.publish(
                        &platform_event.sandbox_id,
                        openshell_core::proto::SandboxStreamEvent {
                            payload: Some(
                                openshell_core::proto::sandbox_stream_event::Payload::Event(
                                    public_platform_event_from_driver(&event),
                                ),
                            ),
                        },
                    );
                }
            }
            None => {}
        }
        Ok(())
    }

    async fn apply_sandbox_update(&self, incoming: DriverSandbox) -> Result<(), String> {
        let _guard = self.sync_lock.lock().await;
        let existing = self
            .store
            .get(Sandbox::object_type(), &incoming.id)
            .await
            .map_err(|e| e.to_string())?;
        self.apply_sandbox_update_locked(incoming, existing).await
    }

    async fn apply_sandbox_update_locked(
        &self,
        incoming: DriverSandbox,
        existing_record: Option<ObjectRecord>,
    ) -> Result<(), String> {
        let existing = existing_record
            .as_ref()
            .map(decode_sandbox_record)
            .transpose()?;
        let previous = existing.clone();

        let mut status = incoming.status.as_ref().map(public_status_from_driver);
        rewrite_user_facing_conditions(
            &mut status,
            existing.as_ref().and_then(|sandbox| sandbox.spec.as_ref()),
        );

        let phase = derive_phase(incoming.status.as_ref());
        let mut sandbox = existing.unwrap_or_else(|| Sandbox {
            id: incoming.id.clone(),
            name: incoming.name.clone(),
            namespace: incoming.namespace.clone(),
            spec: None,
            status: None,
            phase: SandboxPhase::Unknown as i32,
            ..Default::default()
        });

        let old_phase = SandboxPhase::try_from(sandbox.phase).unwrap_or(SandboxPhase::Unknown);
        if old_phase != phase {
            info!(
                sandbox_id = %incoming.id,
                sandbox_name = %incoming.name,
                old_phase = ?old_phase,
                new_phase = ?phase,
                "Sandbox phase changed"
            );
        }

        if phase == SandboxPhase::Error
            && let Some(ref status) = status
        {
            for condition in &status.conditions {
                if condition.r#type == "Ready"
                    && condition.status.eq_ignore_ascii_case("false")
                    && is_terminal_failure_reason(&condition.reason)
                {
                    warn!(
                        sandbox_id = %incoming.id,
                        sandbox_name = %incoming.name,
                        reason = %condition.reason,
                        message = %condition.message,
                        "Sandbox failed to become ready"
                    );
                }
            }
        }

        sandbox.name = incoming.name;
        sandbox.namespace = incoming.namespace;
        sandbox.status = status;
        sandbox.phase = phase as i32;

        // Backfill: if the supervisor session registered before the driver
        // reported this sandbox, the registry's own callback flips to
        // Ready only if the store already holds the record (it didn't
        // back then). This path catches that race by re-checking on every
        // driver snapshot. Only promotes from non-terminal, non-Ready
        // phases so we never mask a driver-reported failure.
        if let Some(registry) = self.supervisor_registry()
            && registry.has_session(&sandbox.id)
        {
            let current = SandboxPhase::try_from(sandbox.phase).unwrap_or(SandboxPhase::Unknown);
            if matches!(current, SandboxPhase::Provisioning | SandboxPhase::Unknown) {
                promote_ready_on_supervisor_session(&mut sandbox);
                info!(
                    sandbox_id = %sandbox.id,
                    sandbox_name = %sandbox.name,
                    old_phase = ?current,
                    new_phase = ?SandboxPhase::Ready,
                    "Sandbox phase changed (backfill via live supervisor session)"
                );
            }
        }

        if previous.as_ref() == Some(&sandbox) {
            return Ok(());
        }

        self.sandbox_index.update_from_sandbox(&sandbox);
        self.store
            .put_message(&sandbox)
            .await
            .map_err(|e| e.to_string())?;
        self.sandbox_watch_bus.notify(&sandbox.id);
        Ok(())
    }

    /// Promote a sandbox's phase to [`SandboxPhase::Ready`] now that its
    /// supervisor session has been established with the gateway.
    ///
    /// This is called from [`crate::supervisor_session::SupervisorSessionRegistry`]
    /// when a `ConnectSupervisor` RPC succeeds. A live supervisor session is
    /// the authoritative "sandbox is usable" signal: it proves the sandbox
    /// has outbound connectivity to the gateway, that mTLS identity matched,
    /// and that the relay plane is ready to service SSH/exec. Compute
    /// drivers (Kubernetes, VM) no longer need to scrape their own liveness.
    ///
    /// No-ops when:
    /// * the sandbox is not in the store (the session registered before the
    ///   driver snapshot landed — the follow-up driver update backfills
    ///   readiness via the `has_session` check in
    ///   `apply_sandbox_update_locked`);
    /// * the sandbox phase is already `Ready` — a reconnect / supersede
    ///   shouldn't churn the record;
    /// * the sandbox phase is `Deleting` or `Error` — those states are
    ///   terminal relative to the compute driver and must not be masked by
    ///   a late-arriving supervisor signal.
    pub async fn mark_sandbox_session_connected(&self, sandbox_id: &str) {
        let _guard = self.sync_lock.lock().await;
        if let Err(err) = self.mark_sandbox_session_connected_locked(sandbox_id).await {
            warn!(%sandbox_id, error = %err, "failed to promote sandbox to Ready on supervisor session");
        }
    }

    async fn mark_sandbox_session_connected_locked(&self, sandbox_id: &str) -> Result<(), String> {
        let Some(record) = self
            .store
            .get(Sandbox::object_type(), sandbox_id)
            .await
            .map_err(|e| e.to_string())?
        else {
            // Session registered before the driver reported the sandbox.
            // The next driver snapshot will call back through
            // `apply_sandbox_update_locked`, which consults
            // `has_session` and promotes on the spot.
            return Ok(());
        };

        let mut sandbox = decode_sandbox_record(&record)?;
        let current = SandboxPhase::try_from(sandbox.phase).unwrap_or(SandboxPhase::Unknown);
        match current {
            SandboxPhase::Ready | SandboxPhase::Deleting | SandboxPhase::Error => {
                return Ok(());
            }
            _ => {}
        }

        promote_ready_on_supervisor_session(&mut sandbox);
        info!(
            %sandbox_id,
            old_phase = ?current,
            new_phase = ?SandboxPhase::Ready,
            "Sandbox phase changed"
        );
        self.sandbox_index.update_from_sandbox(&sandbox);
        self.store
            .put_message(&sandbox)
            .await
            .map_err(|e| e.to_string())?;
        self.sandbox_watch_bus.notify(&sandbox.id);
        Ok(())
    }

    /// Demote a sandbox from `Ready` back to `Provisioning` when its
    /// supervisor session ends. No-op if the sandbox isn't Ready (already
    /// Provisioning, Deleting, or Error).
    pub async fn mark_sandbox_session_disconnected(&self, sandbox_id: &str) {
        let _guard = self.sync_lock.lock().await;
        if let Err(err) = self
            .mark_sandbox_session_disconnected_locked(sandbox_id)
            .await
        {
            warn!(%sandbox_id, error = %err, "failed to demote sandbox on supervisor session end");
        }
    }

    async fn mark_sandbox_session_disconnected_locked(
        &self,
        sandbox_id: &str,
    ) -> Result<(), String> {
        let Some(record) = self
            .store
            .get(Sandbox::object_type(), sandbox_id)
            .await
            .map_err(|e| e.to_string())?
        else {
            return Ok(());
        };

        let mut sandbox = decode_sandbox_record(&record)?;
        let current = SandboxPhase::try_from(sandbox.phase).unwrap_or(SandboxPhase::Unknown);
        if current != SandboxPhase::Ready {
            return Ok(());
        }

        demote_ready_on_supervisor_session(&mut sandbox);
        info!(
            %sandbox_id,
            old_phase = ?current,
            new_phase = ?SandboxPhase::Provisioning,
            "Sandbox phase changed"
        );
        self.sandbox_index.update_from_sandbox(&sandbox);
        self.store
            .put_message(&sandbox)
            .await
            .map_err(|e| e.to_string())?;
        self.sandbox_watch_bus.notify(&sandbox.id);
        Ok(())
    }

    async fn apply_deleted(&self, sandbox_id: &str) -> Result<(), String> {
        let _guard = self.sync_lock.lock().await;
        self.apply_deleted_locked(sandbox_id).await
    }

    async fn apply_deleted_locked(&self, sandbox_id: &str) -> Result<(), String> {
        let _ = self
            .store
            .delete(Sandbox::object_type(), sandbox_id)
            .await
            .map_err(|e| e.to_string())?;
        self.sandbox_index.remove_sandbox(sandbox_id);
        self.sandbox_watch_bus.notify(sandbox_id);
        self.cleanup_sandbox_state(sandbox_id);
        Ok(())
    }

    fn cleanup_sandbox_state(&self, sandbox_id: &str) {
        self.tracing_log_bus.remove(sandbox_id);
        self.tracing_log_bus.platform_event_bus.remove(sandbox_id);
        self.sandbox_watch_bus.remove(sandbox_id);
    }

    async fn reconcile_snapshot_sandbox(
        &self,
        snapshot: DriverSandbox,
        sweep_started_at_ms: i64,
    ) -> Result<(), String> {
        let _guard = self.sync_lock.lock().await;
        let Some(existing) = self
            .store
            .get(Sandbox::object_type(), &snapshot.id)
            .await
            .map_err(|e| e.to_string())?
        else {
            return Ok(());
        };

        if existing.updated_at_ms > sweep_started_at_ms {
            return Ok(());
        }

        let Some(current) = self
            .get_driver_sandbox(&snapshot.id, &snapshot.name)
            .await?
        else {
            return Ok(());
        };

        self.apply_sandbox_update_locked(current, Some(existing))
            .await
    }

    async fn prune_missing_sandbox(
        &self,
        record: ObjectRecord,
        sweep_started_at_ms: i64,
        grace_ms: i64,
    ) -> Result<(), String> {
        let _guard = self.sync_lock.lock().await;
        let Some(current_record) = self
            .store
            .get(Sandbox::object_type(), &record.id)
            .await
            .map_err(|e| e.to_string())?
        else {
            return Ok(());
        };

        if current_record.updated_at_ms > sweep_started_at_ms {
            return Ok(());
        }

        let sandbox = decode_sandbox_record(&current_record)?;
        let age_ms = current_time_ms().saturating_sub(current_record.created_at_ms);
        if age_ms < grace_ms {
            return Ok(());
        }

        if let Some(current) = self.get_driver_sandbox(&sandbox.id, &sandbox.name).await? {
            return self
                .apply_sandbox_update_locked(current, Some(current_record))
                .await;
        }

        info!(
            sandbox_id = %sandbox.id,
            sandbox_name = %sandbox.name,
            age_secs = age_ms / 1000,
            "Removing sandbox from store after it disappeared from the compute driver snapshot"
        );
        self.apply_deleted_locked(&sandbox.id).await
    }

    async fn get_driver_sandbox(
        &self,
        sandbox_id: &str,
        sandbox_name: &str,
    ) -> Result<Option<DriverSandbox>, String> {
        match self
            .driver
            .get_sandbox(Request::new(GetSandboxRequest {
                sandbox_id: sandbox_id.to_string(),
                sandbox_name: sandbox_name.to_string(),
            }))
            .await
        {
            Ok(response) => Ok(response.into_inner().sandbox),
            Err(status) if status.code() == Code::NotFound => Ok(None),
            Err(status) => Err(status.to_string()),
        }
    }
}

fn driver_sandbox_from_public(sandbox: &Sandbox) -> DriverSandbox {
    DriverSandbox {
        id: sandbox.id.clone(),
        name: sandbox.name.clone(),
        namespace: sandbox.namespace.clone(),
        spec: sandbox.spec.as_ref().map(driver_sandbox_spec_from_public),
        status: sandbox
            .status
            .as_ref()
            .map(|status| driver_status_from_public(status, sandbox.phase)),
    }
}

fn driver_sandbox_spec_from_public(spec: &SandboxSpec) -> DriverSandboxSpec {
    DriverSandboxSpec {
        log_level: spec.log_level.clone(),
        environment: spec.environment.clone(),
        template: spec
            .template
            .as_ref()
            .map(driver_sandbox_template_from_public),
        gpu: spec.gpu,
    }
}

fn driver_sandbox_template_from_public(template: &SandboxTemplate) -> DriverSandboxTemplate {
    DriverSandboxTemplate {
        image: template.image.clone(),
        agent_socket_path: template.agent_socket.clone(),
        labels: template.labels.clone(),
        environment: template.environment.clone(),
        resources: extract_typed_resources(&template.resources),
        platform_config: build_platform_config(template),
    }
}

/// Extract typed CPU/memory quantities from the public `resources` Struct.
///
/// The public API exposes resources as an untyped `google.protobuf.Struct`
/// with the Kubernetes limits/requests shape. We pull out the well-known
/// keys into the typed `DriverResourceRequirements` message.
fn extract_typed_resources(
    resources: &Option<prost_types::Struct>,
) -> Option<DriverResourceRequirements> {
    let s = resources.as_ref()?;

    fn get_quantity(s: &prost_types::Struct, section: &str, key: &str) -> String {
        s.fields
            .get(section)
            .and_then(|v| match v.kind.as_ref() {
                Some(prost_types::value::Kind::StructValue(inner)) => inner.fields.get(key),
                _ => None,
            })
            .and_then(|v| match v.kind.as_ref() {
                Some(prost_types::value::Kind::StringValue(val)) => Some(val.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    let req = DriverResourceRequirements {
        cpu_request: get_quantity(s, "requests", "cpu"),
        cpu_limit: get_quantity(s, "limits", "cpu"),
        memory_request: get_quantity(s, "requests", "memory"),
        memory_limit: get_quantity(s, "limits", "memory"),
    };

    // Return None when all fields are empty so drivers can distinguish
    // "no resource requirements" from "zero requirements".
    if req.cpu_request.is_empty()
        && req.cpu_limit.is_empty()
        && req.memory_request.is_empty()
        && req.memory_limit.is_empty()
    {
        None
    } else {
        Some(req)
    }
}

/// Build the opaque `platform_config` Struct from platform-specific public
/// template fields (runtime_class_name, annotations, volume_claim_templates)
/// plus any resource fields beyond CPU/memory.
fn build_platform_config(template: &SandboxTemplate) -> Option<prost_types::Struct> {
    use prost_types::{Struct, Value, value::Kind};

    let mut fields = std::collections::BTreeMap::new();

    if !template.runtime_class_name.is_empty() {
        fields.insert(
            "runtime_class_name".to_string(),
            Value {
                kind: Some(Kind::StringValue(template.runtime_class_name.clone())),
            },
        );
    }

    if !template.annotations.is_empty() {
        let annotation_fields = template
            .annotations
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    Value {
                        kind: Some(Kind::StringValue(v.clone())),
                    },
                )
            })
            .collect();
        fields.insert(
            "annotations".to_string(),
            Value {
                kind: Some(Kind::StructValue(Struct {
                    fields: annotation_fields,
                })),
            },
        );
    }

    // Pass through the raw volume_claim_templates Struct as a nested value.
    if let Some(ref vct) = template.volume_claim_templates {
        fields.insert(
            "volume_claim_templates".to_string(),
            Value {
                kind: Some(Kind::StructValue(vct.clone())),
            },
        );
    }

    // Pass through any non-cpu/memory resource fields from the original
    // resources Struct so the driver can handle GPU limits, custom resources,
    // etc. that don't map to the typed DriverResourceRequirements.
    if let Some(ref res) = template.resources {
        fields.insert(
            "resources_raw".to_string(),
            Value {
                kind: Some(Kind::StructValue(res.clone())),
            },
        );
    }

    if fields.is_empty() {
        None
    } else {
        Some(Struct { fields })
    }
}

fn driver_status_from_public(status: &SandboxStatus, phase: i32) -> DriverSandboxStatus {
    DriverSandboxStatus {
        sandbox_name: status.sandbox_name.clone(),
        instance_id: status.agent_pod.clone(),
        agent_fd: status.agent_fd.clone(),
        sandbox_fd: status.sandbox_fd.clone(),
        conditions: status
            .conditions
            .iter()
            .map(driver_condition_from_public)
            .collect(),
        deleting: SandboxPhase::try_from(phase) == Ok(SandboxPhase::Deleting),
    }
}

fn driver_condition_from_public(condition: &SandboxCondition) -> DriverCondition {
    DriverCondition {
        r#type: condition.r#type.clone(),
        status: condition.status.clone(),
        reason: condition.reason.clone(),
        message: condition.message.clone(),
        last_transition_time: condition.last_transition_time.clone(),
    }
}

impl ObjectType for Sandbox {
    fn object_type() -> &'static str {
        "sandbox"
    }
}

impl ObjectId for Sandbox {
    fn object_id(&self) -> &str {
        &self.id
    }
}

impl ObjectName for Sandbox {
    fn object_name(&self) -> &str {
        &self.name
    }
}

fn compute_error_from_status(status: Status) -> ComputeError {
    match status.code() {
        Code::AlreadyExists => ComputeError::AlreadyExists,
        Code::FailedPrecondition => ComputeError::Precondition(status.message().to_string()),
        _ => ComputeError::Message(status.message().to_string()),
    }
}

fn current_time_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(i64::MAX)
}

fn decode_sandbox_record(record: &ObjectRecord) -> Result<Sandbox, String> {
    Sandbox::decode(record.payload.as_slice()).map_err(|e| e.to_string())
}

fn public_status_from_driver(status: &DriverSandboxStatus) -> SandboxStatus {
    SandboxStatus {
        sandbox_name: status.sandbox_name.clone(),
        agent_pod: status.instance_id.clone(),
        agent_fd: status.agent_fd.clone(),
        sandbox_fd: status.sandbox_fd.clone(),
        conditions: status
            .conditions
            .iter()
            .map(public_condition_from_driver)
            .collect(),
    }
}

fn public_condition_from_driver(condition: &DriverCondition) -> SandboxCondition {
    SandboxCondition {
        r#type: condition.r#type.clone(),
        status: condition.status.clone(),
        reason: condition.reason.clone(),
        message: condition.message.clone(),
        last_transition_time: condition.last_transition_time.clone(),
    }
}

fn public_platform_event_from_driver(event: &DriverPlatformEvent) -> PlatformEvent {
    PlatformEvent {
        timestamp_ms: event.timestamp_ms,
        source: event.source.clone(),
        r#type: event.r#type.clone(),
        reason: event.reason.clone(),
        message: event.message.clone(),
        metadata: event.metadata.clone(),
    }
}

fn derive_phase(status: Option<&DriverSandboxStatus>) -> SandboxPhase {
    if let Some(status) = status {
        if status.deleting {
            return SandboxPhase::Deleting;
        }

        for condition in &status.conditions {
            if condition.r#type == "Ready" {
                return if condition.status.eq_ignore_ascii_case("true") {
                    SandboxPhase::Ready
                } else if condition.status.eq_ignore_ascii_case("false") {
                    if is_terminal_failure_reason(&condition.reason) {
                        SandboxPhase::Error
                    } else {
                        SandboxPhase::Provisioning
                    }
                } else {
                    SandboxPhase::Provisioning
                };
            }
        }
        return SandboxPhase::Provisioning;
    }

    SandboxPhase::Unknown
}

fn rewrite_user_facing_conditions(status: &mut Option<SandboxStatus>, spec: Option<&SandboxSpec>) {
    let gpu_requested = spec.is_some_and(|sandbox_spec| sandbox_spec.gpu);
    if !gpu_requested {
        return;
    }

    if let Some(status) = status {
        for condition in &mut status.conditions {
            if condition.r#type == "Ready"
                && condition.status.eq_ignore_ascii_case("false")
                && condition.reason.eq_ignore_ascii_case("Unschedulable")
            {
                condition.message = "GPU sandbox could not be scheduled on the active gateway. Another GPU sandbox may already be using the available GPU, or the gateway may not currently be able to satisfy GPU placement. Please refer to documentation and use `openshell doctor` commands to inspect GPU support and gateway configuration.".to_string();
            }
        }
    }
}

fn is_terminal_failure_reason(reason: &str) -> bool {
    let reason = reason.to_ascii_lowercase();
    let transient_reasons = ["reconcilererror", "dependenciesnotready", "starting"];
    !transient_reasons.contains(&reason.as_str())
}

/// Reason string written onto the Ready condition when a supervisor session
/// is the trigger for the transition. Kept short and grep-able — the CLI
/// surfaces condition reasons in the sandbox provisioning spinner on
/// timeout. Matches the other reason labels (`Starting`, `Unschedulable`).
pub(crate) const SUPERVISOR_CONNECTED_REASON: &str = "SupervisorConnected";
pub(crate) const SUPERVISOR_DISCONNECTED_REASON: &str = "SupervisorDisconnected";

/// Bridge between [`crate::supervisor_session::SupervisorSessionRegistry`]
/// and [`ComputeRuntime`]. Holds a cloned [`ComputeRuntime`] rather than
/// an [`Arc`]-weak because `ComputeRuntime` is already cheap-to-clone
/// (every non-scalar field is already `Arc`-wrapped internally).
///
/// The cycle with the registry is broken on the runtime side: its own
/// `supervisor_sessions` handle is a [`std::sync::Weak`].
struct ComputeSessionObserver {
    compute: ComputeRuntime,
}

impl crate::supervisor_session::SupervisorSessionObserver for ComputeSessionObserver {
    fn on_session_connected(&self, sandbox_id: String) {
        let compute = self.compute.clone();
        tokio::spawn(async move {
            compute.mark_sandbox_session_connected(&sandbox_id).await;
        });
    }

    fn on_session_disconnected(&self, sandbox_id: String) {
        let compute = self.compute.clone();
        tokio::spawn(async move {
            compute.mark_sandbox_session_disconnected(&sandbox_id).await;
        });
    }
}

/// Flip `sandbox.phase` to `Ready` and update its `Ready` condition to
/// reflect a live supervisor session. Preserves all non-`Ready` conditions
/// the driver may have already reported (e.g. `VmRunning`).
fn promote_ready_on_supervisor_session(sandbox: &mut Sandbox) {
    sandbox.phase = SandboxPhase::Ready as i32;
    let status = sandbox.status.get_or_insert_with(SandboxStatus::default);
    upsert_ready_condition(
        status,
        "True",
        SUPERVISOR_CONNECTED_REASON,
        "Supervisor session established; relay plane is ready",
    );
}

/// Flip `sandbox.phase` back to `Provisioning` after the supervisor session
/// ends. Signals the CLI / TUI that the sandbox is no longer reachable
/// without killing the underlying driver resource, which may recover on a
/// supervisor reconnect.
fn demote_ready_on_supervisor_session(sandbox: &mut Sandbox) {
    sandbox.phase = SandboxPhase::Provisioning as i32;
    let status = sandbox.status.get_or_insert_with(SandboxStatus::default);
    upsert_ready_condition(
        status,
        "False",
        SUPERVISOR_DISCONNECTED_REASON,
        "Supervisor session ended; waiting for the sandbox to reconnect",
    );
}

fn upsert_ready_condition(
    status: &mut SandboxStatus,
    ready_status: &str,
    reason: &str,
    message: &str,
) {
    if let Some(existing) = status.conditions.iter_mut().find(|c| c.r#type == "Ready") {
        existing.status = ready_status.to_string();
        existing.reason = reason.to_string();
        existing.message = message.to_string();
        existing.last_transition_time = String::new();
    } else {
        status.conditions.push(SandboxCondition {
            r#type: "Ready".to_string(),
            status: ready_status.to_string(),
            reason: reason.to_string(),
            message: message.to_string(),
            last_transition_time: String::new(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use openshell_core::proto::compute::v1::{
        CreateSandboxResponse, DeleteSandboxResponse, GetCapabilitiesResponse, GetSandboxRequest,
        GetSandboxResponse, StopSandboxRequest, StopSandboxResponse, ValidateSandboxCreateResponse,
    };
    use std::sync::Arc;

    #[derive(Debug, Default)]
    struct TestDriver {
        listed_sandboxes: Vec<DriverSandbox>,
        current_sandboxes: Vec<DriverSandbox>,
    }

    #[tonic::async_trait]
    impl ComputeDriver for TestDriver {
        type WatchSandboxesStream = DriverWatchStream;

        async fn get_capabilities(
            &self,
            _request: Request<GetCapabilitiesRequest>,
        ) -> Result<tonic::Response<GetCapabilitiesResponse>, Status> {
            Ok(tonic::Response::new(GetCapabilitiesResponse {
                driver_name: "test-driver".to_string(),
                driver_version: "test".to_string(),
                default_image: "openshell/sandbox:test".to_string(),
                supports_gpu: true,
            }))
        }

        async fn validate_sandbox_create(
            &self,
            _request: Request<ValidateSandboxCreateRequest>,
        ) -> Result<tonic::Response<ValidateSandboxCreateResponse>, Status> {
            Ok(tonic::Response::new(ValidateSandboxCreateResponse {}))
        }

        async fn get_sandbox(
            &self,
            request: Request<GetSandboxRequest>,
        ) -> Result<tonic::Response<GetSandboxResponse>, Status> {
            let request = request.into_inner();
            let current = if self.current_sandboxes.is_empty() {
                &self.listed_sandboxes
            } else {
                &self.current_sandboxes
            };
            let sandbox = current
                .iter()
                .find(|sandbox| {
                    sandbox.name == request.sandbox_name
                        && (request.sandbox_id.is_empty() || sandbox.id == request.sandbox_id)
                })
                .cloned()
                .ok_or_else(|| Status::not_found("sandbox not found"))?;

            if !request.sandbox_id.is_empty() && request.sandbox_id != sandbox.id {
                return Err(Status::failed_precondition(
                    "sandbox_id did not match the fetched sandbox",
                ));
            }

            Ok(tonic::Response::new(GetSandboxResponse {
                sandbox: Some(sandbox),
            }))
        }

        async fn list_sandboxes(
            &self,
            _request: Request<ListSandboxesRequest>,
        ) -> Result<
            tonic::Response<openshell_core::proto::compute::v1::ListSandboxesResponse>,
            Status,
        > {
            Ok(tonic::Response::new(
                openshell_core::proto::compute::v1::ListSandboxesResponse {
                    sandboxes: self.listed_sandboxes.clone(),
                },
            ))
        }

        async fn create_sandbox(
            &self,
            _request: Request<CreateSandboxRequest>,
        ) -> Result<tonic::Response<CreateSandboxResponse>, Status> {
            Ok(tonic::Response::new(CreateSandboxResponse {}))
        }

        async fn stop_sandbox(
            &self,
            _request: Request<StopSandboxRequest>,
        ) -> Result<tonic::Response<StopSandboxResponse>, Status> {
            Ok(tonic::Response::new(StopSandboxResponse {}))
        }

        async fn delete_sandbox(
            &self,
            _request: Request<DeleteSandboxRequest>,
        ) -> Result<tonic::Response<DeleteSandboxResponse>, Status> {
            Ok(tonic::Response::new(DeleteSandboxResponse {
                deleted: true,
            }))
        }

        async fn watch_sandboxes(
            &self,
            _request: Request<WatchSandboxesRequest>,
        ) -> Result<tonic::Response<Self::WatchSandboxesStream>, Status> {
            Ok(tonic::Response::new(Box::pin(stream::empty())))
        }
    }

    async fn test_runtime(driver: SharedComputeDriver) -> ComputeRuntime {
        let store = Arc::new(Store::connect("sqlite::memory:").await.unwrap());
        ComputeRuntime {
            driver,
            _driver_process: None,
            default_image: "openshell/sandbox:test".to_string(),
            store,
            sandbox_index: SandboxIndex::new(),
            sandbox_watch_bus: SandboxWatchBus::new(),
            tracing_log_bus: TracingLogBus::new(),
            sync_lock: Arc::new(Mutex::new(())),
            supervisor_sessions: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    fn sandbox_record(id: &str, name: &str, phase: SandboxPhase) -> Sandbox {
        Sandbox {
            id: id.to_string(),
            name: name.to_string(),
            namespace: "default".to_string(),
            phase: phase as i32,
            ..Default::default()
        }
    }

    fn make_driver_condition(reason: &str, message: &str) -> DriverCondition {
        DriverCondition {
            r#type: "Ready".to_string(),
            status: "False".to_string(),
            reason: reason.to_string(),
            message: message.to_string(),
            last_transition_time: String::new(),
        }
    }

    fn make_driver_status(condition: DriverCondition) -> DriverSandboxStatus {
        DriverSandboxStatus {
            sandbox_name: "test".to_string(),
            instance_id: "test-pod".to_string(),
            agent_fd: String::new(),
            sandbox_fd: String::new(),
            conditions: vec![condition],
            deleting: false,
        }
    }

    #[test]
    fn terminal_failure_treats_unknown_reasons_as_terminal() {
        let terminal_cases = [
            ("Failed", "Something went wrong"),
            ("CrashLoopBackOff", "Container keeps crashing"),
            ("ImagePullBackOff", "Failed to pull image"),
            ("ErrImagePull", "Error pulling image"),
            ("Unschedulable", "No nodes match"),
            ("SomeOtherReason", "Any other reason is terminal"),
        ];

        for (reason, message) in terminal_cases {
            assert!(
                is_terminal_failure_reason(reason),
                "Expected terminal failure for reason={reason}, message={message}"
            );
        }
    }

    #[test]
    fn terminal_failure_ignores_transient_reasons() {
        let transient_cases = [
            (
                "ReconcilerError",
                "Error seen: failed to update pod: Operation cannot be fulfilled",
            ),
            ("reconcilererror", "lowercase also works"),
            ("RECONCILERERROR", "uppercase also works"),
            (
                "DependenciesNotReady",
                "Pod exists with phase: Pending; Service Exists",
            ),
            ("dependenciesnotready", "lowercase also works"),
            ("Starting", "VM is starting"),
        ];

        for (reason, message) in transient_cases {
            assert!(
                !is_terminal_failure_reason(reason),
                "Expected transient (non-terminal) for reason={reason}, message={message}"
            );
        }
    }

    #[test]
    fn derive_phase_returns_unknown_without_status() {
        assert_eq!(derive_phase(None), SandboxPhase::Unknown);
    }

    #[test]
    fn derive_phase_returns_deleting_when_driver_marks_deleting() {
        let status = DriverSandboxStatus {
            deleting: true,
            ..make_driver_status(make_driver_condition(
                "DependenciesNotReady",
                "Pod still pending",
            ))
        };

        assert_eq!(derive_phase(Some(&status)), SandboxPhase::Deleting);
    }

    #[test]
    fn derive_phase_returns_provisioning_for_transient_conditions() {
        let transient_conditions = [
            ("ReconcilerError", "Error seen: failed to update pod"),
            (
                "DependenciesNotReady",
                "Pod exists with phase: Pending; Service Exists",
            ),
            ("Starting", "VM is starting"),
        ];

        for (reason, message) in transient_conditions {
            let status = make_driver_status(make_driver_condition(reason, message));
            assert_eq!(
                derive_phase(Some(&status)),
                SandboxPhase::Provisioning,
                "Expected Provisioning for transient reason={reason}"
            );
        }
    }

    #[test]
    fn derive_phase_returns_error_for_terminal_ready_false() {
        let status = make_driver_status(make_driver_condition(
            "ImagePullBackOff",
            "Failed to pull image",
        ));

        assert_eq!(derive_phase(Some(&status)), SandboxPhase::Error);
    }

    #[test]
    fn derive_phase_returns_ready_for_ready_true() {
        let status = DriverSandboxStatus {
            conditions: vec![DriverCondition {
                r#type: "Ready".to_string(),
                status: "True".to_string(),
                reason: "DependenciesReady".to_string(),
                message: "Pod is Ready; Service Exists".to_string(),
                last_transition_time: String::new(),
            }],
            ..make_driver_status(make_driver_condition("", ""))
        };

        assert_eq!(derive_phase(Some(&status)), SandboxPhase::Ready);
    }

    #[test]
    fn rewrite_user_facing_conditions_rewrites_gpu_unschedulable_message() {
        let mut status = Some(SandboxStatus {
            sandbox_name: "test".to_string(),
            agent_pod: "test-pod".to_string(),
            agent_fd: String::new(),
            sandbox_fd: String::new(),
            conditions: vec![SandboxCondition {
                r#type: "Ready".to_string(),
                status: "False".to_string(),
                reason: "Unschedulable".to_string(),
                message: "0/1 nodes are available: 1 Insufficient nvidia.com/gpu.".to_string(),
                last_transition_time: String::new(),
            }],
        });

        rewrite_user_facing_conditions(
            &mut status,
            Some(&SandboxSpec {
                gpu: true,
                ..Default::default()
            }),
        );

        let message = &status.unwrap().conditions[0].message;
        assert_eq!(
            message,
            "GPU sandbox could not be scheduled on the active gateway. Another GPU sandbox may already be using the available GPU, or the gateway may not currently be able to satisfy GPU placement. Please refer to documentation and use `openshell doctor` commands to inspect GPU support and gateway configuration."
        );
    }

    #[test]
    fn rewrite_user_facing_conditions_leaves_non_gpu_unschedulable_message_unchanged() {
        let original = "0/1 nodes are available: 1 Insufficient cpu.";
        let mut status = Some(SandboxStatus {
            sandbox_name: "test".to_string(),
            agent_pod: "test-pod".to_string(),
            agent_fd: String::new(),
            sandbox_fd: String::new(),
            conditions: vec![SandboxCondition {
                r#type: "Ready".to_string(),
                status: "False".to_string(),
                reason: "Unschedulable".to_string(),
                message: original.to_string(),
                last_transition_time: String::new(),
            }],
        });

        rewrite_user_facing_conditions(
            &mut status,
            Some(&SandboxSpec {
                gpu: false,
                ..Default::default()
            }),
        );

        assert_eq!(status.unwrap().conditions[0].message, original);
    }

    #[test]
    fn compute_error_from_status_preserves_driver_status_codes() {
        assert!(matches!(
            compute_error_from_status(Status::already_exists("sandbox already exists")),
            ComputeError::AlreadyExists
        ));

        assert!(matches!(
            compute_error_from_status(Status::failed_precondition("sandbox agent pod IP is not available")),
            ComputeError::Precondition(message) if message == "sandbox agent pod IP is not available"
        ));
    }

    #[tokio::test]
    async fn apply_sandbox_update_allows_delete_failures_to_recover() {
        let runtime = test_runtime(Arc::new(TestDriver::default())).await;
        let sandbox = sandbox_record("sb-1", "sandbox-a", SandboxPhase::Deleting);
        runtime.store.put_message(&sandbox).await.unwrap();

        runtime
            .apply_sandbox_update(DriverSandbox {
                id: "sb-1".to_string(),
                name: "sandbox-a".to_string(),
                namespace: "default".to_string(),
                spec: None,
                status: Some(DriverSandboxStatus {
                    sandbox_name: "sandbox-a".to_string(),
                    instance_id: "agent-pod".to_string(),
                    agent_fd: String::new(),
                    sandbox_fd: String::new(),
                    conditions: vec![DriverCondition {
                        r#type: "Ready".to_string(),
                        status: "True".to_string(),
                        reason: "DependenciesReady".to_string(),
                        message: "Pod is Ready".to_string(),
                        last_transition_time: String::new(),
                    }],
                    deleting: false,
                }),
            })
            .await
            .unwrap();

        let stored = runtime
            .store
            .get_message::<Sandbox>("sb-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SandboxPhase::try_from(stored.phase).unwrap(),
            SandboxPhase::Ready
        );
    }

    #[tokio::test]
    async fn reconcile_store_with_backend_applies_driver_snapshot() {
        let runtime = test_runtime(Arc::new(TestDriver {
            listed_sandboxes: vec![DriverSandbox {
                id: "sb-1".to_string(),
                name: "sandbox-a".to_string(),
                namespace: "default".to_string(),
                spec: None,
                status: Some(DriverSandboxStatus {
                    sandbox_name: "sandbox-a".to_string(),
                    instance_id: "agent-pod".to_string(),
                    agent_fd: String::new(),
                    sandbox_fd: String::new(),
                    conditions: vec![DriverCondition {
                        r#type: "Ready".to_string(),
                        status: "False".to_string(),
                        reason: "DependenciesNotReady".to_string(),
                        message: "Pod is Pending".to_string(),
                        last_transition_time: String::new(),
                    }],
                    deleting: false,
                }),
            }],
            current_sandboxes: vec![DriverSandbox {
                id: "sb-1".to_string(),
                name: "sandbox-a".to_string(),
                namespace: "default".to_string(),
                spec: None,
                status: Some(DriverSandboxStatus {
                    sandbox_name: "sandbox-a".to_string(),
                    instance_id: "agent-pod".to_string(),
                    agent_fd: String::new(),
                    sandbox_fd: String::new(),
                    conditions: vec![DriverCondition {
                        r#type: "Ready".to_string(),
                        status: "True".to_string(),
                        reason: "DependenciesReady".to_string(),
                        message: "Pod is Ready".to_string(),
                        last_transition_time: String::new(),
                    }],
                    deleting: false,
                }),
            }],
            ..Default::default()
        }))
        .await;

        let sandbox = Sandbox {
            spec: Some(SandboxSpec {
                gpu: true,
                ..Default::default()
            }),
            ..sandbox_record("sb-1", "sandbox-a", SandboxPhase::Provisioning)
        };
        runtime.store.put_message(&sandbox).await.unwrap();
        runtime.sandbox_index.update_from_sandbox(&sandbox);

        runtime
            .reconcile_store_with_backend(Duration::ZERO)
            .await
            .unwrap();

        let stored = runtime
            .store
            .get_message::<Sandbox>("sb-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SandboxPhase::try_from(stored.phase).unwrap(),
            SandboxPhase::Ready
        );
        assert!(stored.spec.as_ref().is_some_and(|spec| spec.gpu));
    }

    #[tokio::test]
    async fn reconcile_store_with_backend_does_not_recreate_missing_record_from_snapshot() {
        let runtime = test_runtime(Arc::new(TestDriver {
            listed_sandboxes: vec![DriverSandbox {
                id: "sb-1".to_string(),
                name: "sandbox-a".to_string(),
                namespace: "default".to_string(),
                spec: None,
                status: Some(make_driver_status(make_driver_condition(
                    "DependenciesNotReady",
                    "Pod exists with phase: Pending; Service Exists",
                ))),
            }],
            current_sandboxes: vec![DriverSandbox {
                id: "sb-1".to_string(),
                name: "sandbox-a".to_string(),
                namespace: "default".to_string(),
                spec: None,
                status: Some(make_driver_status(DriverCondition {
                    r#type: "Ready".to_string(),
                    status: "True".to_string(),
                    reason: "DependenciesReady".to_string(),
                    message: "Pod is Ready".to_string(),
                    last_transition_time: String::new(),
                })),
            }],
            ..Default::default()
        }))
        .await;

        runtime
            .reconcile_store_with_backend(Duration::ZERO)
            .await
            .unwrap();

        assert!(
            runtime
                .store
                .get_message::<Sandbox>("sb-1")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn reconcile_store_with_backend_rechecks_driver_before_pruning() {
        let runtime = test_runtime(Arc::new(TestDriver {
            current_sandboxes: vec![DriverSandbox {
                id: "sb-1".to_string(),
                name: "sandbox-a".to_string(),
                namespace: "default".to_string(),
                spec: None,
                status: Some(DriverSandboxStatus {
                    sandbox_name: "sandbox-a".to_string(),
                    instance_id: "agent-pod".to_string(),
                    agent_fd: String::new(),
                    sandbox_fd: String::new(),
                    conditions: vec![DriverCondition {
                        r#type: "Ready".to_string(),
                        status: "True".to_string(),
                        reason: "DependenciesReady".to_string(),
                        message: "Pod is Ready".to_string(),
                        last_transition_time: String::new(),
                    }],
                    deleting: false,
                }),
            }],
            ..Default::default()
        }))
        .await;

        let sandbox = sandbox_record("sb-1", "sandbox-a", SandboxPhase::Provisioning);
        runtime.store.put_message(&sandbox).await.unwrap();
        runtime.sandbox_index.update_from_sandbox(&sandbox);

        runtime
            .reconcile_store_with_backend(Duration::ZERO)
            .await
            .unwrap();

        let stored = runtime
            .store
            .get_message::<Sandbox>("sb-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SandboxPhase::try_from(stored.phase).unwrap(),
            SandboxPhase::Ready
        );
    }

    #[tokio::test]
    async fn reconcile_store_with_backend_removes_stale_provisioning_records() {
        let runtime = test_runtime(Arc::new(TestDriver::default())).await;
        let sandbox = sandbox_record("sb-1", "sandbox-a", SandboxPhase::Provisioning);
        runtime.store.put_message(&sandbox).await.unwrap();
        runtime.sandbox_index.update_from_sandbox(&sandbox);

        let mut watch_rx = runtime.sandbox_watch_bus.subscribe(&sandbox.id);

        runtime
            .reconcile_store_with_backend(Duration::ZERO)
            .await
            .unwrap();

        assert!(
            runtime
                .store
                .get_message::<Sandbox>(&sandbox.id)
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            runtime
                .sandbox_index
                .sandbox_id_for_sandbox_name(&sandbox.name)
                .is_none()
        );
        let _ = watch_rx.try_recv();
        assert!(matches!(
            watch_rx.try_recv(),
            Err(tokio::sync::broadcast::error::TryRecvError::Closed)
        ));
    }

    // ── Gateway-side readiness promotion via supervisor sessions ─────────

    #[tokio::test]
    async fn mark_session_connected_promotes_provisioning_to_ready() {
        let runtime = test_runtime(Arc::new(TestDriver::default())).await;
        let sandbox = sandbox_record("sb-1", "sandbox-a", SandboxPhase::Provisioning);
        runtime.store.put_message(&sandbox).await.unwrap();

        runtime.mark_sandbox_session_connected("sb-1").await;

        let stored = runtime
            .store
            .get_message::<Sandbox>("sb-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SandboxPhase::try_from(stored.phase).unwrap(),
            SandboxPhase::Ready
        );
        let status = stored.status.expect("status should be set");
        let ready = status
            .conditions
            .iter()
            .find(|c| c.r#type == "Ready")
            .expect("Ready condition should be present");
        assert_eq!(ready.status, "True");
        assert_eq!(ready.reason, SUPERVISOR_CONNECTED_REASON);
    }

    #[tokio::test]
    async fn mark_session_connected_is_noop_when_absent() {
        // Session-before-record race: the callback must not create rows.
        let runtime = test_runtime(Arc::new(TestDriver::default())).await;
        runtime.mark_sandbox_session_connected("sb-missing").await;
        assert!(
            runtime
                .store
                .get_message::<Sandbox>("sb-missing")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn mark_session_connected_is_noop_when_deleting() {
        let runtime = test_runtime(Arc::new(TestDriver::default())).await;
        let sandbox = sandbox_record("sb-1", "sandbox-a", SandboxPhase::Deleting);
        runtime.store.put_message(&sandbox).await.unwrap();

        runtime.mark_sandbox_session_connected("sb-1").await;

        let stored = runtime
            .store
            .get_message::<Sandbox>("sb-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SandboxPhase::try_from(stored.phase).unwrap(),
            SandboxPhase::Deleting
        );
    }

    #[tokio::test]
    async fn mark_session_connected_is_noop_when_error() {
        let runtime = test_runtime(Arc::new(TestDriver::default())).await;
        let sandbox = sandbox_record("sb-1", "sandbox-a", SandboxPhase::Error);
        runtime.store.put_message(&sandbox).await.unwrap();

        runtime.mark_sandbox_session_connected("sb-1").await;

        let stored = runtime
            .store
            .get_message::<Sandbox>("sb-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SandboxPhase::try_from(stored.phase).unwrap(),
            SandboxPhase::Error
        );
    }

    #[tokio::test]
    async fn mark_session_connected_is_idempotent_when_already_ready() {
        let runtime = test_runtime(Arc::new(TestDriver::default())).await;
        let mut sandbox = sandbox_record("sb-1", "sandbox-a", SandboxPhase::Provisioning);
        runtime.store.put_message(&sandbox).await.unwrap();

        // First call promotes; second must be a no-op without churning
        // the record.
        runtime.mark_sandbox_session_connected("sb-1").await;
        runtime.mark_sandbox_session_connected("sb-1").await;

        let stored = runtime
            .store
            .get_message::<Sandbox>("sb-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SandboxPhase::try_from(stored.phase).unwrap(),
            SandboxPhase::Ready
        );
        sandbox.phase = SandboxPhase::Ready as i32;
        // Only one Ready condition — no duplicates accumulated.
        let ready_count = stored
            .status
            .as_ref()
            .unwrap()
            .conditions
            .iter()
            .filter(|c| c.r#type == "Ready")
            .count();
        assert_eq!(ready_count, 1);
    }

    #[tokio::test]
    async fn mark_session_disconnected_reverts_ready_to_provisioning() {
        let runtime = test_runtime(Arc::new(TestDriver::default())).await;
        let mut sandbox = sandbox_record("sb-1", "sandbox-a", SandboxPhase::Ready);
        sandbox.status = Some(SandboxStatus {
            sandbox_name: "sandbox-a".to_string(),
            agent_pod: "vm-1".to_string(),
            agent_fd: String::new(),
            sandbox_fd: String::new(),
            conditions: vec![SandboxCondition {
                r#type: "Ready".to_string(),
                status: "True".to_string(),
                reason: SUPERVISOR_CONNECTED_REASON.to_string(),
                message: "ok".to_string(),
                last_transition_time: String::new(),
            }],
        });
        runtime.store.put_message(&sandbox).await.unwrap();

        runtime.mark_sandbox_session_disconnected("sb-1").await;

        let stored = runtime
            .store
            .get_message::<Sandbox>("sb-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SandboxPhase::try_from(stored.phase).unwrap(),
            SandboxPhase::Provisioning
        );
        let ready = stored
            .status
            .unwrap()
            .conditions
            .into_iter()
            .find(|c| c.r#type == "Ready")
            .expect("Ready condition should remain");
        assert_eq!(ready.status, "False");
        assert_eq!(ready.reason, SUPERVISOR_DISCONNECTED_REASON);
    }

    #[tokio::test]
    async fn mark_session_disconnected_is_noop_when_not_ready() {
        let runtime = test_runtime(Arc::new(TestDriver::default())).await;
        let sandbox = sandbox_record("sb-1", "sandbox-a", SandboxPhase::Provisioning);
        runtime.store.put_message(&sandbox).await.unwrap();

        runtime.mark_sandbox_session_disconnected("sb-1").await;

        let stored = runtime
            .store
            .get_message::<Sandbox>("sb-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SandboxPhase::try_from(stored.phase).unwrap(),
            SandboxPhase::Provisioning
        );
    }

    #[tokio::test]
    async fn apply_sandbox_update_backfills_ready_when_session_already_live() {
        // Covers the register-before-store race: the supervisor's
        // ConnectSupervisor landed before the driver reported the
        // sandbox, so the initial callback found no row. The next
        // driver snapshot must check the registry and promote on the
        // spot.
        let runtime = test_runtime(Arc::new(TestDriver::default())).await;
        let registry = Arc::new(crate::supervisor_session::SupervisorSessionRegistry::new());
        *runtime.supervisor_sessions.lock().unwrap() = Some(Arc::downgrade(&registry));

        // Simulate a live session without a persisted sandbox.
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let (shutdown_tx, _shutdown_rx) = tokio::sync::oneshot::channel();
        registry.register("sb-1".to_string(), "session-1".to_string(), tx, shutdown_tx);

        // Driver now reports the sandbox as transient / provisioning.
        let driver_sandbox = DriverSandbox {
            id: "sb-1".to_string(),
            name: "sandbox-a".to_string(),
            namespace: "default".to_string(),
            status: Some(make_driver_status(make_driver_condition(
                "Starting",
                "VM is starting",
            ))),
            ..Default::default()
        };

        runtime.apply_sandbox_update(driver_sandbox).await.unwrap();

        let stored = runtime
            .store
            .get_message::<Sandbox>("sb-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SandboxPhase::try_from(stored.phase).unwrap(),
            SandboxPhase::Ready
        );
        let ready = stored
            .status
            .unwrap()
            .conditions
            .into_iter()
            .find(|c| c.r#type == "Ready")
            .expect("Ready condition should be set");
        assert_eq!(ready.reason, SUPERVISOR_CONNECTED_REASON);
    }

    #[tokio::test]
    async fn apply_sandbox_update_does_not_backfill_when_no_session() {
        // Mirror of the test above, but without a live session: phase
        // must follow the driver's reported condition.
        let runtime = test_runtime(Arc::new(TestDriver::default())).await;
        let registry = Arc::new(crate::supervisor_session::SupervisorSessionRegistry::new());
        *runtime.supervisor_sessions.lock().unwrap() = Some(Arc::downgrade(&registry));

        let driver_sandbox = DriverSandbox {
            id: "sb-1".to_string(),
            name: "sandbox-a".to_string(),
            namespace: "default".to_string(),
            status: Some(make_driver_status(make_driver_condition(
                "Starting",
                "VM is starting",
            ))),
            ..Default::default()
        };

        runtime.apply_sandbox_update(driver_sandbox).await.unwrap();

        let stored = runtime
            .store
            .get_message::<Sandbox>("sb-1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            SandboxPhase::try_from(stored.phase).unwrap(),
            SandboxPhase::Provisioning
        );
    }
}
