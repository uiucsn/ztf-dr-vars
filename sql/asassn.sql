SET max_bytes_before_external_group_by = 2000000000;

SELECT
    asassn_name,
    asassn_type,
    arr.1 AS ztf_oid,
    arr.2 AS filter,
    arr.3 AS mjd,
    arr.4 AS mag,
    arr.5 AS magerr
FROM
(
    SELECT
        asassn_name,
        any(type) AS asassn_type,
        arraySort((arr) -> arr.3, groupArray((ztf_oid, filter, mjd, mag, magerr))) AS arr,
        any(mean_vmag) AS mean_vmag,
        any(amplitude) AS amplitude
    FROM
    (
        SELECT
            oid AS ztf_oid,
            filter,
            mjd,
            mag,
            magerr,
            asassn_name,
            type,
            mean_vmag,
            amplitude
        FROM ztf.dr3
        INNER JOIN
        (
            SELECT
                asassn_name,
                type,
                mean_vmag,
                amplitude,
                arrayJoin(h3kRing(h3index10, toUInt8(ceil((1. / 3600.) / h3EdgeAngle(10))))) AS h3index10,
                radeg AS ra_asassn,
                dedeg AS dec_asassn
            FROM asassn_var.asassn_var_meta
            WHERE (class_probability > 0.9)
        ) AS asassn USING (h3index10)
        WHERE (catflags = 0) AND (greatCircleAngle(ra, dec, ra_asassn, dec_asassn) <= (1. / 3600.))
    )
    GROUP BY asassn_name
    HAVING (NOT (abs(avgIf(mag, filter = 1) - mean_vmag) > 1.0 + amplitude))
)
INTO OUTFILE '/tmp/ztf-asassn.csv'
FORMAT CSV
