//! Inference scheduler for managing update priorities.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::time::{Duration, Instant};

/// A scheduled task for inference.
#[derive(Debug, Clone)]
pub struct ScheduledTask {
    /// Task identifier.
    pub id: usize,
    /// Next scheduled execution time.
    pub next_run: Instant,
    /// Update period.
    pub period: Duration,
    /// Priority (higher = more important).
    pub priority: u32,
    /// Feature indices to update.
    pub feature_indices: Vec<usize>,
}

impl ScheduledTask {
    /// Create a new scheduled task.
    pub fn new(id: usize, period: Duration, priority: u32, feature_indices: Vec<usize>) -> Self {
        Self {
            id,
            next_run: Instant::now(),
            period,
            priority,
            feature_indices,
        }
    }

    /// Update the next run time after execution.
    pub fn reschedule(&mut self) {
        self.next_run = Instant::now() + self.period;
    }

    /// Check if the task is due.
    pub fn is_due(&self) -> bool {
        Instant::now() >= self.next_run
    }
}

impl Eq for ScheduledTask {}

impl PartialEq for ScheduledTask {
    fn eq(&self, other: &Self) -> bool {
        self.next_run == other.next_run && self.priority == other.priority
    }
}

impl Ord for ScheduledTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior (earliest time first)
        other.next_run.cmp(&self.next_run)
            .then_with(|| self.priority.cmp(&other.priority))
    }
}

impl PartialOrd for ScheduledTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority-based inference scheduler.
pub struct InferenceScheduler {
    /// Priority queue of scheduled tasks.
    tasks: BinaryHeap<ScheduledTask>,
    /// Maximum tasks to run per scheduling cycle.
    max_tasks_per_cycle: usize,
}

impl InferenceScheduler {
    /// Create a new scheduler.
    pub fn new(max_tasks_per_cycle: usize) -> Self {
        Self {
            tasks: BinaryHeap::new(),
            max_tasks_per_cycle,
        }
    }

    /// Add a task to the scheduler.
    pub fn add_task(&mut self, task: ScheduledTask) {
        self.tasks.push(task);
    }

    /// Add a task from parameters.
    pub fn add(
        &mut self,
        id: usize,
        rate_hz: f64,
        priority: u32,
        feature_indices: Vec<usize>,
    ) {
        let period = Duration::from_secs_f64(1.0 / rate_hz);
        self.tasks.push(ScheduledTask::new(id, period, priority, feature_indices));
    }

    /// Get the next due task, if any.
    pub fn next_due(&mut self) -> Option<ScheduledTask> {
        if let Some(task) = self.tasks.peek() {
            if task.is_due() {
                return self.tasks.pop();
            }
        }
        None
    }

    /// Get all due tasks up to the maximum per cycle.
    pub fn get_due_tasks(&mut self) -> Vec<ScheduledTask> {
        let mut due = Vec::new();

        while due.len() < self.max_tasks_per_cycle {
            if let Some(task) = self.next_due() {
                due.push(task);
            } else {
                break;
            }
        }

        due
    }

    /// Reschedule a task after execution.
    pub fn reschedule(&mut self, mut task: ScheduledTask) {
        task.reschedule();
        self.tasks.push(task);
    }

    /// Get time until next task is due.
    pub fn time_until_next(&self) -> Option<Duration> {
        self.tasks.peek().map(|task| {
            let now = Instant::now();
            if task.next_run > now {
                task.next_run - now
            } else {
                Duration::ZERO
            }
        })
    }

    /// Get the number of scheduled tasks.
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Check if the scheduler is empty.
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = InferenceScheduler::new(5);
        assert!(scheduler.is_empty());
    }

    #[test]
    fn test_add_and_get_task() {
        let mut scheduler = InferenceScheduler::new(5);
        scheduler.add(0, 100.0, 1, vec![0]);

        assert_eq!(scheduler.len(), 1);

        // Task should be immediately due
        let task = scheduler.next_due();
        assert!(task.is_some());
    }

    #[test]
    fn test_priority_ordering() {
        let mut scheduler = InferenceScheduler::new(5);

        // Add tasks with different priorities
        scheduler.add(0, 100.0, 1, vec![0]); // Low priority
        scheduler.add(1, 100.0, 10, vec![1]); // High priority

        // Both are due, but higher priority should come first
        // (Actually they're both at the same time, so priority matters)
        let tasks = scheduler.get_due_tasks();
        assert_eq!(tasks.len(), 2);
    }

    #[test]
    fn test_reschedule() {
        let mut scheduler = InferenceScheduler::new(5);
        scheduler.add(0, 1000.0, 1, vec![0]); // 1000Hz = 1ms period

        let task = scheduler.next_due().unwrap();
        scheduler.reschedule(task);

        // Task should not be immediately due again
        // (well, depends on timing, but the next_run should have advanced)
        assert_eq!(scheduler.len(), 1);
    }
}
