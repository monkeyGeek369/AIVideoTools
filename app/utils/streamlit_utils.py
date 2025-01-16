from streamlit.runtime import get_instance
from streamlit.runtime.state import SessionState
from streamlit.runtime.session_manager import ActiveSessionInfo


def get_all_sessions() -> list[ActiveSessionInfo]:
    """get all active sessions"""
    runtime = get_instance()
    if runtime is not None:
        session_info = runtime._session_mgr.list_active_sessions()
        return session_info
    return None

def get_all_active_session_states() ->list[SessionState]:
    """get all active session states"""
    sessions = get_all_sessions()
    if not sessions:
        return []
    
    return [s.session.session_state for s in sessions]

def get_all_active_session_task_paths() -> list[str]:
    """get all active session task paths"""
    session_states = get_all_active_session_states()
    if not session_states:
        return []
    return [state['task_path'] for state in session_states if len(state) >0 and state['task_path']]
