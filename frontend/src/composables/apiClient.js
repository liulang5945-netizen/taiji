export const API_BASE = import.meta.env.DEV
  ? ''
  : `${window.location.protocol}//${window.location.hostname}:8000`

export async function authFetch(url, options = {}) {
  const token = localStorage.getItem('jwt_token') || ''
  const headers = new Headers(options.headers || {})
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }
  const response = await fetch(url, { ...options, headers })
  if (response.status === 401) {
    localStorage.removeItem('jwt_token')
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('taiji-auth-expired', {
        detail: { message: 'JWT token 缺失或已过期，请重新登录' },
      }))
    }
  }
  return response
}
