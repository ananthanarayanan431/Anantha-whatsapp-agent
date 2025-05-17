
from datetime import datetime
from typing import Dict, List, Optional, Tuple


from anantha.core.schedules import (
    MONDAY_SCHEDULE,
    TUESDAY_SCHEDULE,
    WEDNESDAY_SCHEDULE,
    THURSDAY_SCHEDULE,
    FRIDAY_SCHEDULE,
    SATURDAY_SCHEDULE,
    SUNDAY_SCHEDULE,
)

class ScheduleContextGenerator:
    "Class to generate context for Anantha's current activity based on schedule."

    SCHEDULES: Dict[int, Dict[str, str]] = {
        0: MONDAY_SCHEDULE,
        1: TUESDAY_SCHEDULE,
        2: WEDNESDAY_SCHEDULE,
        3: THURSDAY_SCHEDULE,
        4: FRIDAY_SCHEDULE,
        5: SATURDAY_SCHEDULE,
        6: SUNDAY_SCHEDULE,
    }

    @staticmethod
    def _parse_time_range(time_range: str) -> Tuple[datetime.time, datetime.time]:
        """Parse a time range string (e.g., '06:00-07:00') into start and end times."""
        start_str, end_str = time_range.split("-")
        start_time = datetime.strptime(start_str, "%H:%M").time()
        end_time = datetime.strptime(end_str, "%H:%M").time()
        return start_time, end_time
    
    @classmethod
    def get_current_activity(cls) -> Optional[str]:
        """Get the current activity based on the current time and day of the week."""

        current_datetime = datetime.now()
        current_time = current_datetime.time()
        current_day = current_datetime.weekday()  # Monday is 0, Sunday is 6

        # Get today's schedule
        today_schedule = cls.SCHEDULES.get(current_day, {})
        
        # Find the current activity
        for time_range, activity in today_schedule.items():
            start_time, end_time = cls._parse_time_range(time_range)
            if start_time <= current_time < end_time:
                return activity
            
            # (e.g., 23:00-06:00) overnight case
            if start_time > end_time:
                if current_time >= start_time or current_time <= end_time:
                    return activity
        
        return None
    
    @classmethod
    def get_schedule_for_day(cls, day: int) -> Dict[str, str]:
        """Get the schedule for a specific day of the week.
        
        Args:
            day (int): The day of the week (0 for Monday, 6 for Sunday).
        
        Returns:
            Dict[str, str]: The schedule for the specified day, or an empty dictionary if the day is invalid.
        """
        return cls.SCHEDULES.get(day, {})