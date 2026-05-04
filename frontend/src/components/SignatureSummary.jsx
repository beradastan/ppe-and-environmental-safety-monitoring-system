import './SignatureSummary.css'

export default function SignatureSummary({ signature }) {
  if (!signature) return null

  const {
    helmet_missing_ids = [],
    vest_missing_ids   = [],
    mask_missing_ids   = [],
    fire_detected      = false,
  } = signature

  const hasViolation =
    helmet_missing_ids.length > 0 ||
    vest_missing_ids.length   > 0 ||
    mask_missing_ids.length   > 0 ||
    fire_detected

  if (!hasViolation) {
    return null
  }

  return (
    <div className="sig-row">
      {helmet_missing_ids.length > 0 && (
        <span className="sig-tag sig-tag--helmet">
          Baretsiz: {helmet_missing_ids.map(id => `#${id}`).join(', ')}
        </span>
      )}
      {vest_missing_ids.length > 0 && (
        <span className="sig-tag sig-tag--vest">
          Yeleksiz: {vest_missing_ids.map(id => `#${id}`).join(', ')}
        </span>
      )}
      {mask_missing_ids.length > 0 && (
        <span className="sig-tag sig-tag--mask">
          Maskesiz: {mask_missing_ids.map(id => `#${id}`).join(', ')}
        </span>
      )}
      {fire_detected && (
        <span className="sig-tag sig-tag--fire">YANGIN TESPİT EDİLDİ</span>
      )}
    </div>
  )
}
