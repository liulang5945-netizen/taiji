function resolveApiBase() {
  if (typeof window === 'undefined') return ''

  const params = new URLSearchParams(window.location.search)
  const hashQuery = window.location.hash.includes('?')
    ? new URLSearchParams(window.location.hash.slice(window.location.hash.indexOf('?') + 1))
    : new URLSearchParams()

  if (params.get('taiji_client') === 'desktop' || hashQuery.get('taiji_client') === 'desktop') {
    return `${window.location.protocol}//${window.location.hostname}:8000`
  }

  if (import.meta.env.DEV) return ''

  const { protocol, hostname, port } = window.location
  if (port === '8000') return ''
  return `${protocol}//${hostname}:8000`
}

export const API_BASE = resolveApiBase()

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
